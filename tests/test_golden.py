import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.golden import (
    Difficulty,
    GoldenCase,
    GoldenDataset,
    GoldenDatasetError,
    compute_stats,
    dump_golden,
    find_issues,
    load_golden,
    save_golden,
)
from app.golden.cli import main
from app.schemas import Category
from tests.golden_fixtures import fixture_case, fixture_dataset, write_json

REPO_GOLDEN = Path(__file__).resolve().parent.parent / "golden" / "v1.json"


@pytest.fixture
def ds_path(tmp_path: Path) -> Path:
    return tmp_path / "v1.json"


def issues_for(path: Path) -> list[str]:
    return [str(i) for i in find_issues(path)]


# --- loading valid files ----------------------------------------------------

def test_repo_golden_file_is_valid_and_has_no_generated_cases():
    dataset = load_golden(REPO_GOLDEN)
    assert dataset.dataset_version == "v1"
    assert dataset.cases == []  # cases are written by hand, never generated


def test_empty_cases_list_loads(ds_path):
    write_json(ds_path, fixture_dataset())
    assert load_golden(ds_path).cases == []


def test_valid_dataset_loads_typed(ds_path):
    write_json(ds_path, fixture_dataset(
        fixture_case(1, expected_category="billing", expected_difficulty="hard", tags=["typo", "very_short"]),
        fixture_case(2),
    ))
    dataset = load_golden(ds_path)
    assert isinstance(dataset, GoldenDataset)
    first = dataset.cases[0]
    assert isinstance(first, GoldenCase)
    assert first.expected_category is Category.BILLING
    assert first.expected_difficulty is Difficulty.HARD
    assert first.tags == ["typo", "very_short"]


def test_tags_are_optional(ds_path):
    case = fixture_case(1)
    del case["tags"]
    write_json(ds_path, fixture_dataset(case))
    assert load_golden(ds_path).cases[0].tags == []


def test_input_is_stored_exactly(ds_path):
    text = "  FIXTURE line one\n\nline three  "
    write_json(ds_path, fixture_dataset(fixture_case(1, input=text)))
    assert load_golden(ds_path).cases[0].input == text


# --- duplicates ---------------------------------------------------------------

def test_duplicate_ids(ds_path):
    write_json(ds_path, fixture_dataset(fixture_case(1), fixture_case(2), fixture_case(2, input="FIXTURE other")))
    assert issues_for(ds_path) == ["case_002: duplicate id (also used by cases[1])"]


def test_duplicate_inputs(ds_path):
    write_json(ds_path, fixture_dataset(
        fixture_case(1, input="FIXTURE same text"),
        fixture_case(2),
        fixture_case(3, input="  FIXTURE same text\n"),  # surrounding whitespace ignored
    ))
    assert issues_for(ds_path) == ["case_003: input is an exact duplicate of case_001"]


def test_inputs_differing_inside_are_not_duplicates(ds_path):
    write_json(ds_path, fixture_dataset(fixture_case(1, input="FIXTURE a b"), fixture_case(2, input="FIXTURE a  b")))
    assert issues_for(ds_path) == []


def test_model_rejects_duplicates_when_built_in_code():
    with pytest.raises(ValidationError, match="duplicate id"):
        GoldenDataset.model_validate(fixture_dataset(fixture_case(1), fixture_case(1, input="FIXTURE x")))


# --- schema violations, each with its case id ------------------------------------

def test_missing_notes(ds_path):
    case = fixture_case(4)
    del case["notes"]
    write_json(ds_path, fixture_dataset(case))
    assert issues_for(ds_path) == ["case_004: notes: Field required"]


@pytest.mark.parametrize("notes", ["", "   "])
def test_blank_notes(ds_path, notes):
    write_json(ds_path, fixture_dataset(fixture_case(1, notes=notes)))
    [issue] = issues_for(ds_path)
    assert issue.startswith("case_001: notes:")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("expected_category", "shipping"),
        ("expected_category", "Billing"),
        ("expected_difficulty", "extreme"),
        ("expected_difficulty", "Hard"),
    ],
)
def test_bad_enum_values(ds_path, field, value):
    write_json(ds_path, fixture_dataset(fixture_case(1, **{field: value})))
    [issue] = issues_for(ds_path)
    assert issue.startswith(f"case_001: {field}: Input should be")


@pytest.mark.parametrize("bad_id", ["case_1", "case_01", "case_0001", "case_000", "CASE_001", "case-001", "001"])
def test_bad_id_format(ds_path, bad_id):
    write_json(ds_path, fixture_dataset(fixture_case(1, id=bad_id)))
    [issue] = issues_for(ds_path)
    assert issue.startswith(f"{bad_id}: id:")


def test_missing_id_reported_by_position(ds_path):
    case = fixture_case(2)
    del case["id"]
    write_json(ds_path, fixture_dataset(fixture_case(1), case))
    assert issues_for(ds_path) == ["cases[1]: id: Field required"]


@pytest.mark.parametrize(
    ("overrides", "field"),
    [
        ({"input": "   "}, "input"),
        ({"ideal_summary": ""}, "ideal_summary"),
        ({"ideal_summary": "x" * 301}, "ideal_summary"),
        ({"tags": ["Typo"]}, "tags.0"),
        ({"tags": ["typo", "typo"]}, "tags"),
        ({"tags": "typo"}, "tags"),
        ({"note": "misspelled key"}, "note"),
    ],
)
def test_other_field_violations(ds_path, overrides, field):
    write_json(ds_path, fixture_dataset(fixture_case(1, **overrides)))
    assert any(i.startswith(f"case_001: {field}:") for i in issues_for(ds_path))


def test_all_errors_reported_not_just_first(ds_path):
    no_notes = fixture_case(2)
    del no_notes["notes"]
    write_json(ds_path, fixture_dataset(
        fixture_case(1, expected_category="shipping"),
        no_notes,
        fixture_case(3, expected_difficulty="extreme", ideal_summary=""),
        fixture_case(3, input="FIXTURE email number 1."),
    ))
    issues = issues_for(ds_path)
    assert [i.split(": ")[0:2] for i in issues] == [
        ["case_001", "expected_category"],
        ["case_002", "notes"],
        ["case_003", "ideal_summary"],
        ["case_003", "expected_difficulty"],
        ["case_003", "duplicate id (also used by cases[2])"],
        ["case_003", "input is an exact duplicate of case_001"],
    ]


def test_load_golden_raises_with_all_issues(ds_path):
    write_json(ds_path, fixture_dataset(fixture_case(1, expected_category="x"), fixture_case(1, input="FIXTURE y")))
    with pytest.raises(GoldenDatasetError) as exc_info:
        load_golden(ds_path)
    assert len(exc_info.value.issues) == 2
    assert "case_001" in str(exc_info.value)


# --- file-level problems ----------------------------------------------------------

@pytest.mark.parametrize("content", ["", "   \n\n"])
def test_empty_file(ds_path, content):
    ds_path.write_text(content, encoding="utf-8")
    assert issues_for(ds_path) == ["dataset: file is empty"]
    with pytest.raises(GoldenDatasetError, match="file is empty"):
        load_golden(ds_path)


def test_invalid_json_reports_line(ds_path):
    ds_path.write_text('{\n  "dataset_version": "v1",\n  "cases": [],\n}\n', encoding="utf-8")
    [issue] = issues_for(ds_path)
    assert issue.startswith("dataset: invalid JSON at line 4")


def test_duplicate_key_in_case(ds_path):
    text = json.dumps(fixture_dataset(fixture_case(1)), indent=2).replace(
        '"notes": "FIXTURE notes 1."', '"notes": "FIXTURE notes 1.",\n "notes": "again"'
    )
    ds_path.write_text(text, encoding="utf-8")
    assert issues_for(ds_path) == ["case_001: key 'notes' appears twice in the same object"]


def test_top_level_must_be_object(ds_path):
    write_json(ds_path, [fixture_case(1)])
    assert issues_for(ds_path) == ["dataset: top level must be a JSON object"]


def test_missing_cases_key(ds_path):
    data = fixture_dataset()
    del data["cases"]
    write_json(ds_path, data)
    assert any("cases: Field required" in i for i in issues_for(ds_path))


def test_header_violations(ds_path):
    write_json(ds_path, fixture_dataset(dataset_version="1", description=""))
    issues = issues_for(ds_path)
    assert any(i.startswith("dataset: dataset_version:") for i in issues)
    assert any(i.startswith("dataset: description:") for i in issues)


def test_version_must_match_file_name(tmp_path):
    path = write_json(tmp_path / "v2.json", fixture_dataset())
    assert issues_for(path) == ["dataset: dataset_version 'v1' does not match file name 'v2'"]


# --- writing --------------------------------------------------------------------

NON_ASCII = [
    "FIXTURE: No puedo iniciar sesión, ¿me ayudan? Gracias — señora Núñez",
    "FIXTURE: Saya tidak boleh log masuk ke akaun saya, tolong bantu 🙏",
]


def test_non_ascii_round_trips_readably(ds_path):
    dataset = GoldenDataset.model_validate(fixture_dataset(
        *(fixture_case(i + 1, input=text, notes=f"FIXTURE nota {i}: acentuación") for i, text in enumerate(NON_ASCII))
    ))
    save_golden(dataset, ds_path)
    raw = ds_path.read_text(encoding="utf-8")
    assert "sesión" in raw and "🙏" in raw and "\\u" not in raw
    assert load_golden(ds_path) == dataset


def test_dump_is_stable_and_in_model_order(ds_path):
    dataset = GoldenDataset.model_validate(fixture_dataset(fixture_case(1, tags=["typo"])))
    text = dump_golden(dataset)
    assert text.endswith("}\n")
    assert '\n  "dataset_version": "v1",' in text  # indent 2
    loaded = json.loads(text)
    assert list(loaded) == ["dataset_version", "created_at", "description", "cases"]
    assert list(loaded["cases"][0]) == list(GoldenCase.model_fields)
    save_golden(dataset, ds_path)
    assert dump_golden(load_golden(ds_path)) == text  # load -> save is a no-op


def test_repo_file_is_already_in_canonical_format():
    assert dump_golden(load_golden(REPO_GOLDEN)) == REPO_GOLDEN.read_text(encoding="utf-8")


# --- id assignment ----------------------------------------------------------------

@pytest.mark.parametrize(
    ("existing", "expected"),
    [([], "case_001"), ([1, 2], "case_003"), ([1, 5], "case_006"), ([5, 1], "case_006"), ([999], "case_1000")],
)
def test_next_case_id(existing, expected):
    dataset = GoldenDataset.model_validate(fixture_dataset(*(fixture_case(n) for n in existing)))
    assert dataset.next_case_id() == expected


# --- add command -------------------------------------------------------------------

def scripted(*answers: str):
    """input() replacement that returns answers in order, then raises EOFError."""
    it: Iterator[str] = iter(answers)

    def _input(prompt: str = "") -> str:
        try:
            return next(it)
        except StopIteration:
            raise EOFError from None

    return _input


ADD_ANSWERS = (
    "FIXTURE line one",
    "FIXTURE línea dos",
    ".",
    "technical",
    "FIXTURE summary.",
    "medium",
    "typo, very_short",
    "FIXTURE why it matters.",
    "y",
)


def test_add_appends_first_case(ds_path, capsys):
    write_json(ds_path, fixture_dataset())
    assert main(["add", str(ds_path)], input_fn=scripted(*ADD_ANSWERS)) == 0
    [case] = load_golden(ds_path).cases
    assert case.id == "case_001"
    assert case.input == "FIXTURE line one\nFIXTURE línea dos"
    assert case.expected_category is Category.TECHNICAL
    assert case.expected_difficulty is Difficulty.MEDIUM
    assert case.tags == ["typo", "very_short"]
    assert "línea" in ds_path.read_text(encoding="utf-8")


def test_add_assigns_next_id_after_highest(ds_path):
    write_json(ds_path, fixture_dataset(fixture_case(1), fixture_case(7)))
    assert main(["add", str(ds_path)], input_fn=scripted(*ADD_ANSWERS)) == 0
    assert [c.id for c in load_golden(ds_path).cases] == ["case_001", "case_007", "case_008"]


def test_add_twice_increments(ds_path):
    write_json(ds_path, fixture_dataset())
    main(["add", str(ds_path)], input_fn=scripted(*ADD_ANSWERS))
    second = ("FIXTURE another email", ".", *ADD_ANSWERS[3:])
    main(["add", str(ds_path)], input_fn=scripted(*second))
    assert [c.id for c in load_golden(ds_path).cases] == ["case_001", "case_002"]


def test_add_reasks_invalid_values(ds_path, capsys):
    write_json(ds_path, fixture_dataset(fixture_case(1)))
    answers = (
        "FIXTURE email number 1.", ".",  # duplicate of case_001: re-asked
        "FIXTURE fresh email", ".",
        "shipping", "account",           # bad category, then good
        "", "FIXTURE summary.",          # empty summary, then good
        "impossible", "hard",
        "Bad Tag", "",                   # bad tag, then none
        "", "FIXTURE notes.",            # notes required
        "yes",
    )
    assert main(["add", str(ds_path)], input_fn=scripted(*answers)) == 0
    out = capsys.readouterr().out
    assert "already has exactly this input" in out
    assert out.count("invalid:") == 6
    new = load_golden(ds_path).cases[-1]
    assert (new.id, new.expected_category, new.expected_difficulty, new.tags) == ("case_002", "account", "hard", [])


@pytest.mark.parametrize("answers", [ADD_ANSWERS[:4], (*ADD_ANSWERS[:-1], "n")])
def test_add_cancel_or_decline_writes_nothing(ds_path, answers):
    write_json(ds_path, fixture_dataset(fixture_case(1)))
    before = ds_path.read_bytes()
    assert main(["add", str(ds_path)], input_fn=scripted(*answers)) == 1
    assert ds_path.read_bytes() == before


def test_add_refuses_invalid_file(ds_path, capsys):
    write_json(ds_path, fixture_dataset(fixture_case(1, notes="")))
    assert main(["add", str(ds_path)], input_fn=scripted(*ADD_ANSWERS)) == 1
    assert "case_001: notes" in capsys.readouterr().err


# --- validate / stats commands -------------------------------------------------------

def test_validate_command_prints_every_error(ds_path, capsys):
    write_json(ds_path, fixture_dataset(fixture_case(1, expected_category="x"), fixture_case(2, notes="")))
    assert main(["validate", str(ds_path)]) == 1
    out = capsys.readouterr().out
    assert "case_001: expected_category" in out and "case_002: notes" in out


def test_validate_command_ok(ds_path, capsys):
    write_json(ds_path, fixture_dataset(fixture_case(1)))
    assert main(["validate", str(ds_path)]) == 0
    assert "OK" in capsys.readouterr().out


def _balanced_cases(hard_per_category: int) -> list[dict]:
    cases, n = [], 0
    for category in Category:
        for i in range(8):
            n += 1
            difficulty = "hard" if i < hard_per_category else "easy"
            cases.append(fixture_case(n, expected_category=category.value, expected_difficulty=difficulty, tags=["typo"] if i == 0 else []))
    return cases


def test_stats_counts_and_no_warnings_when_balanced():
    stats = compute_stats(GoldenDataset.model_validate(fixture_dataset(*_balanced_cases(hard_per_category=2))))
    assert stats.total == 32
    assert set(stats.by_category.values()) == {8}
    assert stats.by_difficulty == {Difficulty.EASY: 24, Difficulty.MEDIUM: 0, Difficulty.HARD: 8}
    assert stats.by_tag == {"typo": 4}
    assert stats.untagged == 28
    assert stats.warnings == []


def test_stats_warns_on_small_category_and_few_hard_cases():
    cases = _balanced_cases(hard_per_category=1)[:-1]  # general has 7; hard = 4/31 = 12.9%
    stats = compute_stats(GoldenDataset.model_validate(fixture_dataset(*cases)))
    assert stats.warnings == [
        "category 'general' has 7 case(s); aim for at least 8",
        "hard cases are 12.9% of the total; aim for at least 15%",
    ]


def test_stats_command_runs(ds_path, capsys):
    write_json(ds_path, fixture_dataset(*_balanced_cases(hard_per_category=0)))
    assert main(["stats", str(ds_path)]) == 0
    out = capsys.readouterr().out
    assert "Total cases: 32" in out and "hard cases are 0.0%" in out
