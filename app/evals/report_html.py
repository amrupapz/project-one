"""Self-contained HTML report for a comparison: no JavaScript, no CDN, inline SVG charts.

Every piece of run data (emails, summaries, notes) is HTML-escaped: golden cases include
injection attempts and must render as text.
"""

import math
from dataclasses import dataclass
from datetime import datetime
from html import escape

from app.evals.compare import Comparison, Delta, Flip
from app.evals.models import CaseResult, RunInfo
from app.evals.store import RunSummary


@dataclass(frozen=True)
class TrendPoint:
    run_id: str
    started_at: datetime
    prompt_version: str
    strict: float | None
    judge: float | None  # None = a gap in the judge chart; judge_gap says why
    judge_gap: str | None = None  # "not judged" or "judge coverage 12/68"


def _judge_gap(r: RunSummary, min_judge_coverage: float) -> str | None:
    if not r.info.judge_enabled:
        return "not judged"
    if r.case_count == 0 or r.judged / r.case_count < min_judge_coverage:
        return f"judge coverage {r.judged}/{r.case_count}"
    return None


def trend_points(
    runs: list[RunSummary], candidate: RunInfo, limit: int, min_judge_coverage: float = 0.95
) -> list[TrendPoint]:
    """The last `limit` full runs on the candidate's dataset, up to and including the
    candidate, oldest first. `runs` is newest first (as list_runs returns it).
    A run without the judge, or with judge coverage below min_judge_coverage, has no
    judge value (a gap); its strict accuracy is still plotted."""
    picked = [
        r
        for r in runs
        if not r.info.is_partial
        and r.info.dataset_sha256 == candidate.dataset_sha256
        and (r.info.started_at, r.info.run_id) <= (candidate.started_at, candidate.run_id)
    ][:limit]
    points = []
    for r in reversed(picked):
        gap = _judge_gap(r, min_judge_coverage)
        points.append(
            TrendPoint(
                run_id=r.info.run_id,
                started_at=r.info.started_at,
                prompt_version=r.info.prompt_version,
                strict=r.strict_accuracy,
                judge=None if gap else r.mean_judge_score,
                judge_gap=gap,
            )
        )
    return points


# --- formatting helpers ----------------------------------------------------------------

def _e(value: object) -> str:
    return escape("" if value is None else str(value), quote=True)


def _pct(v: float | None) -> str:
    return "–" if v is None else f"{v:.1%}"


def _num(v: float | None, fmt: str) -> str:
    return "–" if v is None else format(v, fmt)


def _delta_cell(d: Delta, fmt: str, scale: float = 1.0, unit: str = "", higher_is_better: bool | None = True) -> str:
    if d.change is None:
        return '<td class="num muted">–</td>'
    value = d.change * scale
    text = ("+" if value >= 0 else "−") + format(abs(value), fmt) + unit
    cls = "num"
    if higher_is_better is not None and round(value, 9) != 0:
        cls += " better" if (value > 0) == higher_is_better else " worse"
    return f'<td class="{cls}">{text}</td>'


# --- SVG line chart ----------------------------------------------------------------------

def _ticks(lo: float, hi: float, steps: tuple[float, ...]) -> list[float]:
    """Round tick values: the smallest step from `steps` giving at most 6 ticks."""
    step = next((st for st in steps if (hi - lo) / st <= 5), steps[-1])
    first = -(-round(lo / step, 9) // 1) * step  # ceil to a multiple of step
    out, v = [], first
    while v <= hi + 1e-9:
        out.append(round(v, 9))
        v += step
    return out


def _x_labels(points: list[TrendPoint]) -> list[str]:
    """Second x-axis line: start time (UTC), with the date if the window spans days,
    and the run id's random suffix if two runs would still look the same."""
    spans_days = len({p.started_at.date() for p in points}) > 1
    labels = [f"{p.started_at:%m-%d %H:%M}" if spans_days else f"{p.started_at:%H:%M}" for p in points]
    keyed = [f"{p.prompt_version} {lab}" for p, lab in zip(points, labels)]
    return [
        f"{lab} ·{p.run_id.rsplit('-', 1)[-1]}" if keyed.count(k) > 1 else lab
        for p, lab, k in zip(points, labels, keyed)
    ]


def _line_chart(
    title: str,
    points: list[TrendPoint],
    attr: str,
    lo: float,
    hi: float,
    fmt,
    highlight: dict[str, str],
    steps: tuple[float, ...],
    caption: str,
) -> str:
    """One series, one y-axis. None values break the line (e.g. --no-judge runs)."""
    width, height = 640, 236
    left, right, top, bottom = 52, 20, 16, 56
    plot_w, plot_h = width - left - right, height - top - bottom
    n = len(points)

    def x(i: int) -> float:
        return left + (plot_w / 2 if n == 1 else i * plot_w / (n - 1))

    def y(v: float) -> float:
        return top + plot_h * (1 - (v - lo) / (hi - lo))

    parts = [
        f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" aria-label="{_e(title)}">',
    ]
    for v in _ticks(lo, hi, steps):  # recessive gridlines + y labels
        parts.append(f'<line class="grid" x1="{left}" x2="{width - right}" y1="{y(v):.1f}" y2="{y(v):.1f}"/>')
        parts.append(f'<text class="axis" x="{left - 8}" y="{y(v) + 4:.1f}" text-anchor="end">{_e(fmt(v))}</text>')

    segments: list[list[str]] = [[]]
    for i, p in enumerate(points):
        v = getattr(p, attr)
        if v is None:
            if segments[-1]:
                segments.append([])
            continue
        segments[-1].append(f"{x(i):.1f},{y(v):.1f}")
    for seg in segments:
        if len(seg) > 1:
            parts.append(f'<polyline class="line" points="{" ".join(seg)}"/>')

    x_labels = _x_labels(points)
    for i, p in enumerate(points):
        label = f"{p.prompt_version} · {p.started_at:%Y-%m-%d %H:%M} UTC"
        anchor = "end" if i == n - 1 and n > 1 else "start" if i == 0 and n > 1 else "middle"
        parts.append(
            f'<text class="axis" x="{x(i):.1f}" y="{height - bottom + 18}" text-anchor="{anchor}">'
            f'<tspan class="xv">{_e(p.prompt_version)}</tspan>'
            f'<tspan x="{x(i):.1f}" dy="14">{_e(x_labels[i])}</tspan></text>'
        )
        v = getattr(p, attr)
        role = highlight.get(p.run_id, "")
        gap = (p.judge_gap if attr == "judge" else None) or "no value"
        tip = f"{p.run_id} ({label}): " + (gap if v is None else fmt(v))
        if role:
            tip += f" — {role}"
        if v is None:
            parts.append(
                f'<g class="pt"><title>{_e(tip)}</title>'
                f'<text class="axis muted" x="{x(i):.1f}" y="{top + plot_h - 4}" text-anchor="{anchor}">{_e(gap)}</text></g>'
            )
            continue
        cls = "dot hl" if role else "dot"
        parts.append(
            f'<g class="pt"><title>{_e(tip)}</title>'
            f'<circle class="hit" cx="{x(i):.1f}" cy="{y(v):.1f}" r="12"/>'
            f'<circle class="{cls}" cx="{x(i):.1f}" cy="{y(v):.1f}" r="{5 if role else 4}"/></g>'
        )
        if role:  # selective direct labels: only baseline and candidate
            parts.append(
                f'<text class="val" x="{x(i):.1f}" y="{y(v) - 12:.1f}" text-anchor="{anchor}">{_e(fmt(v))} {_e(role)}</text>'
            )
    parts.append("</svg>")
    return (
        f'<figure><figcaption>{_e(title)}</figcaption>{"".join(parts)}'
        f'<p class="caption muted">{_e(caption)}</p></figure>'
    )


def _trend_section(points: list[TrendPoint], cmp: Comparison) -> str:
    if len(points) < 2:
        return "<section><h2>Trend</h2><p class=\"muted\">Fewer than two full runs on this dataset; no trend yet.</p></section>"
    highlight = {cmp.baseline.run_id: "baseline", cmp.candidate.run_id: "candidate"}
    stricts = [p.strict for p in points if p.strict is not None]
    s_lo = max(0.0, math.floor(min(stricts) * 20) / 20 - 0.05) if stricts else 0.0  # down to 5%, pad 5%
    judges = [p.judge for p in points if p.judge is not None]
    j_lo = max(1.0, math.floor(min(judges) * 2) / 2 - 0.5) if judges else 1.0
    s_caption = (
        "Y-axis starts at 0% (full scale)." if s_lo == 0
        else f"Y-axis starts at {s_lo:.0%}, not 0%, so differences look larger than on a full scale."
    )
    charts = _line_chart("Strict accuracy, last full runs", points, "strict", s_lo, 1.0,
                         lambda v: f"{v:.1%}", highlight, (0.01, 0.02, 0.05, 0.1, 0.2, 0.25), s_caption)
    if judges:
        j_caption = (
            "Y-axis starts at 1 (full 1–5 scale)." if j_lo == 1
            else f"Y-axis starts at {j_lo:.2f}, not 1 (the scale is 1–5), so differences look larger."
        )
        gaps = [p for p in points if p.judge is None]
        if gaps:
            j_caption += " Gaps: " + "; ".join(f"{p.prompt_version} {p.started_at:%H:%M} ({p.judge_gap})" for p in gaps) + "."
        charts += _line_chart("Mean judge score (1–5), last full runs", points, "judge", j_lo, 5.0,
                              lambda v: f"{v:.2f}", highlight, (0.25, 0.5, 1.0), j_caption)
    else:
        charts += '<p class="muted">No judged runs in this window; no judge-score trend.</p>'
    rows = "".join(
        f"<tr><td><code>{_e(p.run_id)}</code></td><td>{_e(p.prompt_version)}</td>"
        f"<td>{p.started_at:%Y-%m-%d %H:%M}</td><td class=\"num\">{_pct(p.strict)}</td>"
        f"<td class=\"num\">{_e(p.judge_gap) if p.judge is None else f'{p.judge:.2f}'}</td></tr>"
        for p in points
    )
    table = (
        "<details><summary>Trend data as a table</summary><table><thead><tr><th>Run</th><th>Prompt</th>"
        "<th>Started (UTC)</th><th class=\"num\">Strict</th><th class=\"num\">Judge</th></tr></thead>"
        f"<tbody>{rows}</tbody></table></details>"
    )
    return f"<section id=\"trend\"><h2>Trend ({len(points)} full runs)</h2><div class=\"charts\">{charts}</div>{table}</section>"


# --- sections ----------------------------------------------------------------------------

def _run_meta(role: str, info: RunInfo) -> str:
    flags = []
    if info.is_partial:
        flags.append(f'<span class="tag warn">PARTIAL: first {info.case_limit} cases</span>')
    if not info.judge_enabled:
        flags.append('<span class="tag warn">NO JUDGE</span>')
    rows = [
        ("Run", f"<code>{_e(info.run_id)}</code> {' '.join(flags)}"),
        ("Started (UTC)", f"{info.started_at:%Y-%m-%d %H:%M:%S}"),
        ("Prompt", f"{_e(info.prompt_version)} <span class=\"muted\">sha256 {_e(info.prompt_file_sha256[:12])}</span>"),
        ("Classifier", _e(info.classifier_model)),
        ("Judge", _e(info.judge_model) if info.judge_enabled else "skipped (--no-judge)"),
        ("Dataset", f"{_e(info.dataset_version)} <span class=\"muted\">sha256 {_e(info.dataset_sha256[:12])}</span>"),
        ("Eval config", f"<span class=\"muted\">sha256 {_e(info.eval_config_sha256[:12])}</span>"),
        ("Notes", _e(info.notes) if info.notes else '<span class="muted">–</span>'),
    ]
    body = "".join(f"<tr><th>{k}</th><td>{v}</td></tr>" for k, v in rows)
    return f'<div class="meta"><h3>{role}</h3><table>{body}</table></div>'


def _scorecard(cmp: Comparison) -> str:
    def row(label: str, b: str, c: str, delta: str) -> str:
        return f"<tr><th>{label}</th><td class=\"num\">{b}</td><td class=\"num\">{c}</td>{delta}</tr>"

    bc, cc = cmp.baseline_card, cmp.candidate_card
    judge_note = "" if cmp.judge_used_in_status else ' <span class="muted">(not used for status)</span>'
    rows = [
        row("Strict accuracy", f"{_pct(cmp.strict.baseline)} <span class=\"muted\">{bc.overall.strict_correct}/{bc.overall.total}</span>",
            f"{_pct(cmp.strict.candidate)} <span class=\"muted\">{cc.overall.strict_correct}/{cc.overall.total}</span>",
            _delta_cell(cmp.strict, ".1f", 100, " pp")),
        row("Acceptable accuracy", f"{_pct(cmp.acceptable.baseline)} <span class=\"muted\">{bc.overall.acceptable_correct}/{bc.overall.total}</span>",
            f"{_pct(cmp.acceptable.candidate)} <span class=\"muted\">{cc.overall.acceptable_correct}/{cc.overall.total}</span>",
            _delta_cell(cmp.acceptable, ".1f", 100, " pp")),
        row(f"Mean judge score{judge_note}",
            f"{_num(cmp.judge_score.baseline, '.2f')} <span class=\"muted\">{cmp.judge_coverage[0].scored}/{cmp.judge_coverage[0].total} scored</span>",
            f"{_num(cmp.judge_score.candidate, '.2f')} <span class=\"muted\">{cmp.judge_coverage[1].scored}/{cmp.judge_coverage[1].total} scored</span>",
            _delta_cell(cmp.judge_score, ".2f")),
        row("Latency p50", f"{_num(cmp.latency_p50_ms.baseline, ',.0f')} ms", f"{_num(cmp.latency_p50_ms.candidate, ',.0f')} ms",
            _delta_cell(cmp.latency_p50_ms, ",.0f", unit=" ms", higher_is_better=False)),
        row("Latency p95", f"{_num(cmp.latency_p95_ms.baseline, ',.0f')} ms", f"{_num(cmp.latency_p95_ms.candidate, ',.0f')} ms",
            _delta_cell(cmp.latency_p95_ms, ",.0f", unit=" ms", higher_is_better=False)),
        row("Tokens in", _num(cmp.input_tokens.baseline, ",.0f"), _num(cmp.input_tokens.candidate, ",.0f"),
            _delta_cell(cmp.input_tokens, ",.0f", higher_is_better=None)),
        row("Tokens out", _num(cmp.output_tokens.baseline, ",.0f"), _num(cmp.output_tokens.candidate, ",.0f"),
            _delta_cell(cmp.output_tokens, ",.0f", higher_is_better=None)),
    ]
    head = '<thead><tr><th></th><th class="num">Baseline</th><th class="num">Candidate</th><th class="num">Delta</th></tr></thead>'
    main = f"<table class=\"score\">{head}<tbody>{''.join(rows)}</tbody></table>"

    def groups(title: str, items) -> str:
        body = "".join(
            f"<tr><th>{_e(k.value)}</th><td class=\"num\">{g.n}</td>"
            f"<td class=\"num\">{_pct(g.strict.baseline)}</td><td class=\"num\">{_pct(g.strict.candidate)}</td>{_delta_cell(g.strict, '.1f', 100, ' pp')}"
            f"<td class=\"num\">{_pct(g.acceptable.baseline)}</td><td class=\"num\">{_pct(g.acceptable.candidate)}</td>{_delta_cell(g.acceptable, '.1f', 100, ' pp')}</tr>"
            for k, g in items
        )
        return (
            f"<table class=\"score\"><thead><tr><th>{title}</th><th class=\"num\">n</th>"
            "<th class=\"num\">Strict base</th><th class=\"num\">Strict cand</th><th class=\"num\">Δ</th>"
            "<th class=\"num\">Accept base</th><th class=\"num\">Accept cand</th><th class=\"num\">Δ</th></tr></thead>"
            f"<tbody>{body}</tbody></table>"
        )

    return (
        "<section><h2>Scorecard</h2>" + main
        + groups("By expected category", cmp.by_category.items())
        + groups("By difficulty", cmp.by_difficulty.items())
        + "</section>"
    )


def _output(c: CaseResult) -> str:
    if c.error is not None:
        return f'<div class="pred err">ERROR</div><div class="sum">{_e(c.error)}</div>'
    return f'<div class="pred">{_e(c.predicted_category.value if c.predicted_category else "–")}</div><div class="sum">{_e(c.predicted_summary)}</div>'


def _flip_table(title: str, flips: list[Flip], kind: str) -> str:
    if not flips:
        return f'<h3>{title} (0)</h3><p class="muted">None.</p>'
    rows = []
    for f in flips:
        acceptable = ", ".join(c.value for c in f.acceptable_categories)
        expected = _e(f.expected_category.value) + (f' <span class="muted">(acceptable: {_e(acceptable)})</span>' if acceptable else "")
        rows.append(
            f'<tr class="{kind}" id="{kind}-{_e(f.case_id)}">'
            f"<td><code>{_e(f.case_id)}</code><div class=\"muted\">{_e(f.candidate.expected_difficulty.value)}</div></td>"
            f'<td><div class="email">{_e(f.input)}</div></td>'
            f"<td><div class=\"pred\">{expected}</div><div class=\"sum\">{_e(f.ideal_summary)}</div></td>"
            f"<td>{_output(f.baseline)}</td><td>{_output(f.candidate)}</td></tr>"
        )
    head = "<thead><tr><th>Case</th><th>Email</th><th>Expected / ideal summary</th><th>Baseline output</th><th>Candidate output</th></tr></thead>"
    return f'<h3>{title} ({len(flips)})</h3><table class="flips">{head}<tbody>{"".join(rows)}</tbody></table>'


STATUS_ICON = {"PASS": "✓", "WARNING": "!", "CRITICAL": "✕"}

CSS = """
:root { color-scheme: light dark; }
.viz-root {
  --surface-1:#fcfcfb; --surface-2:#f3f2ee; --text-primary:#0b0b0b; --text-secondary:#52514e;
  --muted:#898781; --grid:#e1e0d9; --border:rgba(11,11,11,0.10); --series-1:#2a78d6;
  --good:#0ca30c; --warning:#fab219; --critical:#d03b3b; --better:#0a7a0a; --worse:#b42828;
}
@media (prefers-color-scheme: dark) {
  .viz-root {
    --surface-1:#1a1a19; --surface-2:#232322; --text-primary:#f0efec; --text-secondary:#c3c2b7;
    --muted:#898781; --grid:#2c2c2a; --border:rgba(255,255,255,0.10); --series-1:#3987e5;
    --better:#4cc24c; --worse:#e66767;
  }
}
* { box-sizing: border-box; }
body { margin:0; }
.viz-root { background:var(--surface-1); color:var(--text-primary); min-height:100vh;
  font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif; padding:28px clamp(16px,4vw,48px); }
h1 { font-size:22px; margin:0 0 4px; } h2 { font-size:17px; margin:32px 0 12px; } h3 { font-size:15px; margin:20px 0 8px; }
.muted { color:var(--muted); } code { font-size:12.5px; }
.status { display:inline-flex; align-items:center; gap:10px; padding:10px 16px; border-radius:8px;
  border:1px solid var(--border); background:var(--surface-2); font-size:17px; font-weight:600; margin:12px 0; }
.status .icon { display:inline-grid; place-items:center; width:24px; height:24px; border-radius:50%; color:#fff; font-size:14px; }
.status.PASS .icon { background:var(--good); } .status.WARNING .icon { background:var(--warning); color:#0b0b0b; }
.status.CRITICAL .icon { background:var(--critical); }
.warnings { border-left:4px solid var(--warning); background:var(--surface-2); padding:10px 14px; margin:12px 0; border-radius:4px; }
.warnings li { margin:2px 0; }
.metas { display:grid; grid-template-columns:repeat(auto-fit,minmax(340px,1fr)); gap:16px; }
.meta table th { text-align:left; color:var(--text-secondary); font-weight:500; padding-right:12px; white-space:nowrap; }
table { border-collapse:collapse; margin:6px 0 14px; }
.score th, .score td, .flips th, .flips td { padding:6px 10px; border-bottom:1px solid var(--grid); vertical-align:top; }
.score thead th, .flips thead th { text-align:left; color:var(--text-secondary); font-weight:500; font-size:12.5px; }
.score tbody th { text-align:left; font-weight:500; }
.num { text-align:right; font-variant-numeric:tabular-nums; }
.better { color:var(--better); } .worse { color:var(--worse); font-weight:600; }
.tag { font-size:11.5px; padding:1px 6px; border-radius:4px; border:1px solid var(--border); margin-left:4px; }
.tag.warn { background:var(--warning); color:#0b0b0b; }
.flips { width:100%; table-layout:fixed; }
.flips td:nth-child(1) { width:90px; } .flips td:nth-child(2) { width:34%; }
.email { white-space:pre-wrap; max-height:180px; overflow:auto; font-size:13px; color:var(--text-secondary); }
.pred { font-weight:600; } .pred.err { color:var(--worse); } .sum { font-size:13px; }
tr.regression td:first-child { border-left:3px solid var(--worse); }
tr.improvement td:first-child { border-left:3px solid var(--better); }
.charts { display:grid; grid-template-columns:repeat(auto-fit,minmax(380px,1fr)); gap:16px; }
figure { margin:0; } figcaption { font-weight:500; margin-bottom:4px; }
.chart { width:100%; height:auto; }
.chart .grid { stroke:var(--grid); stroke-width:1; }
.chart .axis { fill:var(--muted); font-size:11px; }
.chart .line { fill:none; stroke:var(--series-1); stroke-width:2; stroke-linejoin:round; }
.chart .dot { fill:var(--series-1); stroke:var(--surface-1); stroke-width:2; }
.chart .dot.hl { stroke:var(--text-primary); stroke-width:2; }
.chart .hit { fill:transparent; }
.chart .pt:hover .dot { r:7; }
.chart .xv { fill:var(--text-secondary); }
.caption { font-size:12px; margin:2px 0 0; }
.chart .val { fill:var(--text-primary); font-size:11.5px; font-weight:500; }
"""


def render_report(cmp: Comparison, trend: list[TrendPoint], *, generated_at: datetime) -> str:
    status = cmp.status.name
    warnings = "".join(f"<li>{_e(w)}</li>" for w in cmp.warnings)
    reasons = "".join(f"<li>{_e(r)}</li>" for r in cmp.status_reasons)
    notes = "".join(f"<li>{_e(n)}</li>" for n in cmp.notes)
    noise = (
        f"With {cmp.shared_cases} shared cases one case is about {cmp.points_per_case:.1f} pp, "
        "so small differences can be noise. Run <code>python -m app.evals repeat</code> to measure it."
    )
    flips = (
        "<section><h2>Flipped cases</h2>"
        f"<p>Strict: {len(cmp.strict_regressions)} regressed, {len(cmp.strict_improvements)} improved. "
        f"Acceptable: {len(cmp.acceptable_regressions)} regressed, {len(cmp.acceptable_improvements)} improved. {noise}</p>"
        + _flip_table("Strict regressions", cmp.strict_regressions, "regression")
        + _flip_table("Strict improvements", cmp.strict_improvements, "improvement")
        + _flip_table("Acceptable regressions", cmp.acceptable_regressions, "regression")
        + _flip_table("Acceptable improvements", cmp.acceptable_improvements, "improvement")
        + "</section>"
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Eval comparison {_e(cmp.candidate.run_id)}: {_e(status)}</title>
<style>{CSS}</style></head>
<body><div class="viz-root">
<h1>Eval comparison: {_e(cmp.baseline.prompt_version)} → {_e(cmp.candidate.prompt_version)}</h1>
<div class="muted">Generated {generated_at:%Y-%m-%d %H:%M:%S %Z} · {cmp.shared_cases} shared cases</div>
<div class="status {status}" role="status"><span class="icon" aria-hidden="true">{STATUS_ICON[status]}</span><span>{_e(cmp.headline)}</span></div>
{f'<ul class="reasons">{reasons}</ul>' if reasons else ''}
{f'<div class="warnings"><strong>Warnings</strong><ul>{warnings}</ul></div>' if warnings else ''}
{f'<ul class="muted">{notes}</ul>' if notes else ''}
<section><h2>Runs</h2><div class="metas">{_run_meta("Baseline", cmp.baseline)}{_run_meta("Candidate", cmp.candidate)}</div></section>
{_scorecard(cmp)}
{flips}
{_trend_section(trend, cmp)}
</div></body></html>
"""
