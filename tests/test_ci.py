"""Tests for the CI regression check (app/ci.py). The API is mocked as in the other eval tests."""

import json
import subprocess
from pathlib import Path

import pytest
import yaml

from app.ci import (
    COMMENT_MARKER,
    EXIT_ERROR,
    MAX_FLIP_ROWS,
    CiOutcome,
    GitRepo,
    PlannedComparison,
    PromptChange,
    PromptResult,
    format_pr_comment,
    main,
    parse_name_status,
    plan_comparisons,
    run_ci,
)
from app.evals.compare import compare_runs
from app.evals.models import EvalRun
from tests.conftest import VALID_PROMPT
from tests.golden_fixtures import fixture_case, fixture_dataset, write_json
from tests.test_evals import RUBRIC, classifier_reply, fake_client, judge_reply, make_case, make_info

BASE_SHA = "0123456789abcdef0123456789abcdef01234567"


# --- changed-prompt detection --------------------------------------------------------------

def test_parse_name_status_keeps_added_and_modified_prompts_only():
    output = "\n".join([
        "M\tprompts/v2.yaml",
        "A\tprompts/v3.yaml",
        "D\tprompts/v1.yaml",           # deleted: nothing to run
        "M\tapp/classify.py",           # not a prompt
        "A\tprompts/notes.md",          # not yaml
        "A\tprompts/v4.yml",            # wrong extension
        "A\tprompts/old/v0.yaml",       # nested: not a prompt file
        "M\tgolden/v1.json",
        "",
    ])
    assert parse_name_status(output) == [
        PromptChange("prompts/v2.yaml", "modified"),
        PromptChange("prompts/v3.yaml", "added"),
    ]


def test_parse_name_status_empty():
    assert parse_name_status("") == []


def test_plan_modified_vs_base_version_and_new_vs_baseline_prompt():
    plans = plan_comparisons(
        [PromptChange("prompts/v1.yaml", "modified"), PromptChange("prompts/v3.yaml", "added")],
        baseline_prompt="prompts/v1.yaml",
    )
    assert plans == [
        PlannedComparison("prompts/v1.yaml", "prompts/v1.yaml", "base version"),
        PlannedComparison("prompts/v3.yaml", "prompts/v1.yaml", "baseline_prompt"),
    ]


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True).stdout.strip()


def test_git_repo_detects_changes_and_reads_base_versions(tmp_path):
    repo = tmp_path / "repo"
    (repo / "prompts").mkdir(parents=True)
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "ci@example.invalid")
    _git(repo, "config", "user.name", "CI test")
    (repo / "prompts" / "v1.yaml").write_text("base v1\n")
    (repo / "prompts" / "v2.yaml").write_text("base v2\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "base")
    base = _git(repo, "rev-parse", "HEAD")

    (repo / "prompts" / "v2.yaml").write_text("pr v2\n")
    (repo / "prompts" / "v3.yaml").write_text("pr v3\n")
    (repo / "prompts" / "v1.yaml").unlink()
    (repo / "README.md").write_text("docs\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "pr")

    git = GitRepo(repo)
    assert parse_name_status(git.changed_files(base)) == [
        PromptChange("prompts/v2.yaml", "modified"),
        PromptChange("prompts/v3.yaml", "added"),
    ]
    assert git.show(base, "prompts/v2.yaml") == b"base v2\n"  # the base version, not the PR's
    with pytest.raises(FileNotFoundError, match="prompts/v3.yaml does not exist on the base commit"):
        git.show(base, "prompts/v3.yaml")
    with pytest.raises(RuntimeError, match="git diff"):
        git.changed_files("not-a-commit")


# --- full CI run with a fake git and a mocked API ---------------------------------------------

class FakeGit:
    def __init__(self, name_status: str, base_files: dict[str, bytes]) -> None:
        self.name_status = name_status
        self.base_files = base_files
        self.shown: list[str] = []

    def changed_files(self, base_sha: str) -> str:
        return self.name_status

    def show(self, base_sha: str, path: str) -> bytes:
        self.shown.append(path)
        if path not in self.base_files:
            raise FileNotFoundError(f"{path} does not exist on the base commit {base_sha[:12]}")
        return self.base_files[path]


def prompt_yaml(version: str, system_prompt: str) -> bytes:
    return yaml.safe_dump({**VALID_PROMPT, "version": version, "system_prompt": system_prompt}).encode()


@pytest.fixture
def workspace(tmp_path):
    """A PR checkout (head_root) with prompts, a fixture dataset and an eval config."""
    head = tmp_path / "head"
    (head / "prompts").mkdir(parents=True)
    (head / "golden").mkdir()
    dataset = write_json(head / "golden" / "v1.json", fixture_dataset(
        *(fixture_case(n, expected_category="billing") for n in range(1, 11))
    ))
    config = head / "eval_config.yaml"
    config.write_text(yaml.safe_dump({
        "judge_model": "anthropic/claude-haiku-4.5", "judge_rubric": RUBRIC,
        "baseline_prompt": "prompts/v1.yaml", "db_path": "results/eval.db",
    }), encoding="utf-8")
    return {"head": head, "dataset": dataset, "config": config, "out": tmp_path / "out"}


@pytest.fixture
def prompt_sensitive_classifier(monkeypatch):
    """Answers billing (correct) unless the system prompt contains WORSE, then account (wrong)."""
    calls: list[dict] = []

    async def handler(kwargs):
        calls.append(kwargs)
        system = kwargs["messages"][0]["content"]
        return classifier_reply("account" if "WORSE" in system else "billing")

    client, _ = fake_client(monkeypatch, handler)
    return client, calls


async def _run(workspace, git, classifier, judge=False, judge_client=None):
    return await run_ci(
        base_sha=BASE_SHA, out_dir=workspace["out"], eval_config_path=workspace["config"],
        dataset_path=workspace["dataset"], judge=judge, git=git, head_root=workspace["head"],
        classifier_client=classifier, judge_client=judge_client,
    )


async def test_no_prompt_changes_skips_the_eval(workspace, prompt_sensitive_classifier, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)  # not needed when nothing runs
    classifier, calls = prompt_sensitive_classifier
    outcome = await _run(workspace, FakeGit("M\tapp/classify.py\nM\tgolden/v1.json\n", {}), classifier)
    assert outcome.results == [] and outcome.exit_code == 0 and calls == []


async def test_modified_prompt_regression_is_critical(workspace, prompt_sensitive_classifier):
    classifier, calls = prompt_sensitive_classifier
    (workspace["head"] / "prompts" / "v2.yaml").write_bytes(prompt_yaml("v2", "Classify. WORSE"))
    git = FakeGit("M\tprompts/v2.yaml\n", {"prompts/v2.yaml": prompt_yaml("v2", "Classify carefully.")})

    outcome = await _run(workspace, git, classifier)
    [result] = outcome.results
    assert result.comparison.strict.baseline == 1.0 and result.comparison.strict.candidate == 0.0
    assert outcome.exit_code == 2  # CRITICAL
    assert result.report_path.is_file() and "<html" in result.report_path.read_text().lower()
    # The base version was materialized under its own file name (version id must match it).
    assert (workspace["out"] / "base" / "prompts" / "v2.yaml").read_bytes() == prompt_yaml("v2", "Classify carefully.")
    assert len(calls) == 20  # 10 cases x 2 runs, judge off
    assert (workspace["out"] / "eval.db").is_file()


async def test_new_prompts_share_one_baseline_run(workspace, prompt_sensitive_classifier):
    classifier, calls = prompt_sensitive_classifier
    for v in ("v3", "v4"):
        (workspace["head"] / "prompts" / f"{v}.yaml").write_bytes(prompt_yaml(v, f"Classify {v}."))
    git = FakeGit("A\tprompts/v3.yaml\nA\tprompts/v4.yaml\n", {"prompts/v1.yaml": prompt_yaml("v1", "Classify v1.")})

    outcome = await _run(workspace, git, classifier)
    assert [r.plan.reason for r in outcome.results] == ["baseline_prompt", "baseline_prompt"]
    assert outcome.exit_code == 0
    assert git.shown == ["prompts/v1.yaml"]  # baseline read from the base commit once...
    assert len(calls) == 30  # ...and run once: 3 runs x 10 cases


async def test_one_broken_prompt_does_not_stop_the_others(workspace, prompt_sensitive_classifier):
    classifier, _ = prompt_sensitive_classifier
    (workspace["head"] / "prompts" / "v2.yaml").write_bytes(prompt_yaml("v2", "Classify."))
    (workspace["head"] / "prompts" / "v3.yaml").write_text("version: v3\n")  # invalid: fields missing
    git = FakeGit("M\tprompts/v2.yaml\nA\tprompts/v3.yaml\n", {
        "prompts/v1.yaml": prompt_yaml("v1", "Classify."), "prompts/v2.yaml": prompt_yaml("v2", "Classify."),
    })
    outcome = await _run(workspace, git, classifier)
    ok, broken = outcome.results
    assert ok.comparison is not None and ok.exit_code == 0
    assert broken.comparison is None and "ValidationError" in broken.error
    assert outcome.exit_code == EXIT_ERROR


async def test_judge_flag_runs_the_judge(workspace, prompt_sensitive_classifier, monkeypatch):
    classifier, _ = prompt_sensitive_classifier
    judge_calls: list = []

    async def judge_handler(kwargs):
        judge_calls.append(kwargs)
        return judge_reply(4)

    judge, _ = fake_client(monkeypatch, judge_handler)
    (workspace["head"] / "prompts" / "v2.yaml").write_bytes(prompt_yaml("v2", "Classify."))
    git = FakeGit("M\tprompts/v2.yaml\n", {"prompts/v2.yaml": prompt_yaml("v2", "Classify!")})
    outcome = await _run(workspace, git, classifier, judge=True, judge_client=judge)
    assert len(judge_calls) == 20
    assert outcome.results[0].comparison.judge_score.candidate == 4.0


async def test_missing_api_key_is_a_clear_error(workspace, monkeypatch, capsys):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    (workspace["head"] / "prompts" / "v2.yaml").write_bytes(prompt_yaml("v2", "Classify."))
    git = FakeGit("M\tprompts/v2.yaml\n", {"prompts/v2.yaml": prompt_yaml("v2", "Classify!")})
    outcome = await _run(workspace, git, classifier=None)
    assert outcome.exit_code == EXIT_ERROR
    assert "OPENROUTER_API_KEY is not set" in outcome.error and "forks" in outcome.error
    assert capsys.readouterr().out.startswith("::error title=Missing secret::")


def test_main_writes_comment_and_never_prints_the_key(workspace, monkeypatch, capsys):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-FIXTURE-SECRET")
    code = main(
        ["--base-sha", BASE_SHA, "--out", str(workspace["out"]), "--eval-config", str(workspace["config"]),
         "--dataset", str(workspace["dataset"])],
        git=FakeGit("M\tapp/llm.py\n", {}),
    )
    assert code == 0
    comment = (workspace["out"] / "comment.md").read_text()
    assert comment.startswith(COMMENT_MARKER) and "eval was skipped" in comment
    captured = capsys.readouterr()
    assert "sk-or-FIXTURE-SECRET" not in captured.out + captured.err + comment


def test_main_git_failure_is_an_error_comment(workspace, capsys):
    class BrokenGit(FakeGit):
        def changed_files(self, base_sha):
            raise RuntimeError("git diff against deadbeef failed: bad revision")

    code = main(["--base-sha", "deadbeef", "--out", str(workspace["out"]), "--eval-config", str(workspace["config"])],
                git=BrokenGit("", {}))
    assert code == EXIT_ERROR
    assert "## ❌ LLM regression check: ERROR" in (workspace["out"] / "comment.md").read_text()


# --- comment formatting ------------------------------------------------------------------------

def _comparison(n_cases: int = 10, regress: int = 0, improve: int = 0, soft: int = 0, **info):
    """Baseline correct on cases[regress:]; candidate wrong on the first `regress` cases
    (the first `soft` of them still acceptable) and newly right on `improve` extra cases."""
    base_cases, cand_cases = [], []
    for i in range(1, n_cases + 1):
        cid = f"case_{i:03d}"
        acceptable = ["billing", "technical"] if i <= soft else []
        if i <= regress:
            base_cases.append(make_case(cid, "billing", "billing", acceptable_categories=acceptable))
            cand_cases.append(make_case(cid, "billing", "technical", acceptable_categories=acceptable))
        elif i <= regress + improve:
            base_cases.append(make_case(cid, "billing", "account"))
            cand_cases.append(make_case(cid, "billing", "billing"))
        else:
            base_cases.append(make_case(cid))
            cand_cases.append(make_case(cid))
    return compare_runs(
        EvalRun(info=make_info("20261125-120000-base", judge_enabled=False, **info), cases=base_cases),
        EvalRun(info=make_info("20261125-120100-cand", judge_enabled=False, **info), cases=cand_cases),
    )


def _result(path="prompts/v2.yaml", reason="base version", baseline=None, **kw):
    return PromptResult(PlannedComparison(path, baseline or path, reason), comparison=_comparison(**kw))


def _comment(outcome, judge=False):
    return format_pr_comment(outcome, judge=judge, dataset="golden/v1.json", base_sha=BASE_SHA,
                             run_url="https://github.com/o/r/actions/runs/1")


def test_comment_marker_status_and_tables():
    text = _comment(CiOutcome([_result(n_cases=20, regress=2, improve=1, soft=1)]))
    assert text.splitlines()[0] == COMMENT_MARKER  # used to find and update the comment on re-push
    assert "## ⚠️ LLM regression check: WARNING" in text  # net 1 of 20 cases lost = -5 pp (warning at 3, critical at 8)
    assert "| Strict accuracy | 90.0% | 85.0% | -5.0 pp |" in text
    assert "| Acceptable accuracy |" in text and "| Mean judge score (not used for status) |" in text
    assert "| case_001 | billing | billing → technical | regressed | – |" in text  # still acceptable
    assert "| case_002 | billing | billing → technical | regressed | regressed |" in text
    assert "| case_003 | billing | account → billing | improved | improved |" in text
    assert "**Strict vs acceptable:**" in text and "one case is about 5.0 pp" in text
    assert "judge **off**" in text and "[workflow run and HTML reports](https://github.com/o/r/actions/runs/1)" in text
    assert "Gate: **CRITICAL** or **ERROR** fails this check; **WARNING** passes." in text
    assert "golden/v1.json" in text and BASE_SHA[:7] in text and BASE_SHA[:8] not in text


@pytest.mark.parametrize(("regress", "label"), [(0, "PASS"), (1, "WARNING"), (2, "CRITICAL")])
def test_comment_status_follows_thresholds(regress, label):
    # 25 cases: one case = 4 pp; warning at 3 pp, critical at 8 pp
    outcome = CiOutcome([_result(n_cases=25, regress=regress)])
    assert outcome.exit_code == {"PASS": 0, "WARNING": 1, "CRITICAL": 2}[label]
    assert f"LLM regression check: {label}" in _comment(outcome)


def test_comment_new_file_names_its_baseline():
    text = _comment(CiOutcome([_result("prompts/v3.yaml", "baseline_prompt", "prompts/v1.yaml")]))
    assert "### `prompts/v3.yaml` vs base `prompts/v1.yaml` (`baseline_prompt`; this is a new file)" in text


def test_comment_multiple_prompts_has_summary_table_and_worst_status():
    outcome = CiOutcome([_result("prompts/v2.yaml", n_cases=25), _result("prompts/v3.yaml", n_cases=25, regress=1)])
    text = _comment(outcome)
    assert "LLM regression check: WARNING" in text
    assert "| `prompts/v2.yaml` | PASS |" in text and "| `prompts/v3.yaml` | WARNING |" in text


def test_comment_caps_flip_rows():
    text = _comment(CiOutcome([_result(n_cases=MAX_FLIP_ROWS + 15, regress=MAX_FLIP_ROWS + 5)]))
    assert text.count("| regressed |") == MAX_FLIP_ROWS
    assert "…and 5 more in the HTML report." in text


def test_comment_for_error_result_and_job_error():
    broken = PromptResult(PlannedComparison("prompts/v3.yaml", "prompts/v1.yaml", "baseline_prompt"),
                          error="ValidationError: ```system_prompt``` Field required")
    text = _comment(CiOutcome([broken]))
    assert "LLM regression check: ERROR" in text and "could not be completed" in text
    assert text.count("```") == 2  # the error's own backticks can't break the code block

    text = _comment(CiOutcome(error="OPENROUTER_API_KEY is not set."))
    assert "LLM regression check: ERROR" in text and "OPENROUTER_API_KEY is not set." in text


def test_comment_skipped_and_judge_on():
    text = _comment(CiOutcome(), judge=True)
    assert "LLM regression check: PASS" in text and "eval was skipped" in text and "judge **on**" in text


def test_comment_shows_partial_and_config_warnings():
    cmp = _comparison()
    text = _comment(CiOutcome([PromptResult(PlannedComparison("prompts/v2.yaml", "prompts/v2.yaml", "base version"),
                                            comparison=cmp)]))
    for w in cmp.warnings:
        assert f"- ⚠️ {w}" in text


def test_comment_is_capped_in_size(monkeypatch):
    monkeypatch.setattr("app.ci.MAX_COMMENT_CHARS", 2000)
    text = _comment(CiOutcome([_result(n_cases=40, regress=20)]))
    assert len(text) <= 2000 and text.endswith("…(truncated; see the HTML reports)\n")


def test_overall_exit_code_is_worst():
    assert CiOutcome().exit_code == 0
    assert CiOutcome([_result(n_cases=25), _result(n_cases=25, regress=2)]).exit_code == 2
    broken = PromptResult(PlannedComparison("prompts/v9.yaml", "prompts/v1.yaml", "baseline_prompt"), error="x")
    assert CiOutcome([_result(n_cases=25, regress=2), broken]).exit_code == EXIT_ERROR
    assert CiOutcome(error="no key").exit_code == EXIT_ERROR
