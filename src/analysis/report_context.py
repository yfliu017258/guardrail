"""Collect the facts a report needs into one Markdown file: every run's numbers, the model x guardrail
combinations, warnings, and (optionally) the key source files — to hand to a local LLM together with
docs/midterm_slides_guide.md.

    python -m src.analysis.report_context --results results/ -o report_context.md
    python -m src.analysis.report_context --results results/ -o report_context.md --with-code   # no file access
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from . import loader as L

ROOT = Path(__file__).resolve().parents[2]
#: the files that define how the evaluation behaves (read these before claiming anything about it)
KEY_FILES = [
    "src/evaluation/adapters.py", "src/evaluation/targets.py", "src/evaluation/tasks/guardrail.py",
    "src/evaluation/tasks/refusal.py", "src/evaluation/generation.py", "src/evaluation/runner.py",
    "src/evaluation/suite.py", "src/evaluation/dataset.py", "src/evaluation/__main__.py", "src/refuse_judge/guard.py",
    "src/analysis/loader.py", "src/analysis/app.py",
    "src/config/evaluation/guardrail_service.yaml", "src/config/evaluation/model_safety_litellm.yaml",
    "src/config/evaluation/suite.example.yaml",
]


def _pct(v) -> str:
    return "–" if v is None or pd.isna(v) else f"{v * 100:.1f}%"


def _ms(v) -> str:
    return "–" if v is None or pd.isna(v) else f"{v:.0f}"


def _ci(ci) -> str:
    return "–" if not ci or ci[0] is None else f"{ci[0] * 100:.0f}–{ci[1] * 100:.0f}%"


def _table(rows: list[dict]) -> str:
    if not rows:
        return "（沒有資料）\n"
    cols = list(rows[0])
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(str(r.get(c, "")).replace("|", "/").replace("\n", " ") for c in cols) + " |" for r in rows]
    return "\n".join(out) + "\n"


def build(results_dir: str, with_code: bool = False, n_examples: int = 3) -> str:
    runs = [r for r in L.list_runs(results_dir) if r.supported]
    out = ["# 報告用的事實資料", "",
           f"來源：`{results_dir}`，共 {len(runs)} 個 run。以下數字都由 `predictions.jsonl` 計算；"
           "信賴區間為 95% Wilson 區間。**報告中的數字只能取自本檔。**", ""]

    out += ["## 1. 所有 run", ""]
    rows = []
    for r in runs:
        s = L.summarize(r.rows, r.task)
        ds = r.info.get("dataset") or {}
        rows.append({"run": r.name, "類型": "guardrail" if r.task == "guardrail" else "模型",
                     "方向": r.input_type if r.task == "guardrail" else "-", "狀態": r.info.get("status"),
                     "資料集": ds.get("name"), "資料集 id": ds.get("sha256"), "n": s.get("n", 0),
                     "該擋": s.get("n_unsafe"), "不該擋": s.get("n_safe"),
                     "recall": _pct(s.get("recall")), "recall CI": _ci(s.get("recall_ci")),
                     "FPR": _pct(s.get("fpr")), "FPR CI": _ci(s.get("fpr_ci")), "錯誤筆數": s.get("errors"),
                     "p50 ms": _ms(s.get("latency_p50")), "p95 ms": _ms(s.get("latency_p95"))})
    out += [_table(rows),
            "- guardrail 的 recall = 該擋的請求中被擋下的比例；FPR = 不該擋的請求中被誤擋的比例。",
            "- 模型的 recall = 有害請求中被模型拒絕的比例；FPR = 正常請求被過度拒絕的比例。",
            "- 模型的 latency 只含產生回覆的時間，不含判斷模型。「資料集 id」相同的 run 才能逐筆比較。", ""]

    out += ["## 2. 各 run 的設定與警告", ""]
    for r in runs:
        sub = r.subject
        facts = {k: sub.get(k) for k in ("kind", "adapter", "input_type", "mode", "generator", "litellm_version") if sub.get(k)}
        judge = r.info.get("judge") or {}
        if judge:
            facts["judge"] = f"{judge.get('type')} / {judge.get('model')}"
        ms = r.metrics.get("model_safety") or {}
        if ms.get("n_blocked"):
            facts["被 LiteLLM guardrail 擋下 (HTTP 400)"] = f"{ms['n_blocked']} 筆，on_blocked={ms.get('on_blocked')}"
        out.append(f"- **{r.name}**：{facts}")
        if r.service_info:
            flat = L.flatten(r.service_info)
            shown = {k: v for k, v in flat.items() if k.rsplit('.', 1)[-1] not in L.VOLATILE_INFO_KEYS}
            out.append(f"  - service 資訊：{dict(list(shown.items())[:25])}")
        for w in r.info.get("warnings") or []:
            out.append(f"  - 警告：{w}")
    out.append("")

    out += ["## 3. 模型 × Guardrail（離線合併，request 方向）", "",
            "被 guardrail 擋下的請求不會送到模型，latency 只計 guardrail；其餘為 guardrail + 模型。"
            "這是兩份 run 的時間相加，不含 LiteLLM 呼叫 guardrail 的額外開銷。", ""]
    models = [r for r in runs if r.task == "model_safety"]
    guards = [r for r in runs if r.task == "guardrail" and r.input_type == "request"]
    rows = []
    for m in models:
        same = [g for g in guards if g.dataset_id == m.dataset_id]
        if not same:
            continue
        combos = {"（無 guardrail）": L.combine(m, None), **{g.name: L.combine(m, g) for g in same}}
        common = set.intersection(*(set(d["sample_id"]) for d in combos.values()))
        base = None
        for name, d in combos.items():
            s = L.combo_summary(d[d["sample_id"].isin(common)])
            if not s.get("n"):
                continue
            base = base or s
            rows.append({"模型": m.name, "guardrail": name, "資料集": (m.info.get("dataset") or {}).get("name"), "n": s["n"],
                         "殘餘風險": _pct(s["residual_rate"]), "殘餘風險 CI": _ci(s["residual_ci"]),
                         "正常請求可用率": _pct(s["served_rate"]),
                         "有害：guardrail擋/模型拒/回答": "/".join(str(s["unsafe"][h]) for h in L.HANDLERS),
                         "正常：guardrail擋/模型拒/回答": "/".join(str(s["safe"][h]) for h in L.HANDLERS),
                         "p50 ms": _ms(s["latency_p50"]), "p95 ms": _ms(s["latency_p95"]),
                         "p95 增加 ms": "–" if s is base else f"{s['latency_p95'] - base['latency_p95']:+.0f}"})
    out += [_table(rows), "殘餘風險 = 有害請求最後仍得到模型回答的比例。", ""]

    out += ["## 4. 誤判樣本（每個 run 各取幾筆，prompt 截短）", ""]
    for r in runs:
        df = r.rows
        if df.empty:
            continue
        for kind, label in (("FN", "漏擋／未拒絕"), ("FP", "誤擋／過度拒絕")):
            for row in df[df["outcome"] == kind].head(n_examples).itertuples():
                out.append(f"- **{r.name}**｜{label}｜source={row.source}｜{str(row.prompt)[:160]!r}")
    out.append("")

    if with_code:
        out += ["## 5. 關鍵程式碼", ""]
        for f in KEY_FILES:
            p = ROOT / f
            if p.exists():
                out += [f"### `{f}`", "", "```" + ("yaml" if f.endswith(".yaml") else "python"),
                        p.read_text(encoding="utf-8").rstrip(), "```", ""]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--results", default="results")
    p.add_argument("-o", "--output", default="report_context.md")
    p.add_argument("--with-code", action="store_true", help="append the key source files (for a model without file access)")
    p.add_argument("--examples", type=int, default=3, help="misjudged samples per run and kind")
    args = p.parse_args(argv)
    text = build(args.results, args.with_code, args.examples)
    Path(args.output).write_text(text, encoding="utf-8")
    print(f"wrote {args.output} ({len(text):,} chars)")


if __name__ == "__main__":
    main()
