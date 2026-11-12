"""Interactive analysis of evaluation results.

    streamlit run src/analysis/app.py -- --results results/

Pages: 總覽 · Run 詳情 · Run 比較 · 模型 × Guardrail · Latency. Reads ``results/<run>/`` only.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:  # `streamlit run` puts the script's folder on sys.path, not the repo root
    sys.path.insert(0, str(ROOT))

from src.analysis import loader as L  # noqa: E402

# categorical slots (validated as a set for colour-vision deficiency); colour follows the entity everywhere
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
HANDLER_COLORS = {"guardrail": BLUE, "model": AQUA, "answered": ORANGE}
RUN_COLORS = [BLUE, ORANGE, AQUA]  # more runs than this -> direct labels / table carry identity
PLOT_CFG = {"displayModeBar": False}


# --------------------------------------------------------------------------- data
def _results_dir() -> str:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--results", default="results")
    args, _ = p.parse_known_args(sys.argv[1:])
    return args.results


@st.cache_resource(show_spinner="讀取 results …")
def _load(results_dir: str, fingerprint: float) -> list[L.Run]:
    runs = L.list_runs(results_dir)
    for r in runs:
        _ = r.rows  # load once; cached until a run changes on disk
    return runs


def _pct(v: float | None, digits: int = 1) -> str:
    return "–" if v is None or pd.isna(v) else f"{v * 100:.{digits}f}%"


def _ms(v: float | None) -> str:
    return "–" if v is None or pd.isna(v) else f"{v:,.0f} ms"


def _ci(ci: tuple[float | None, float | None]) -> str:
    return "" if ci[0] is None else f"95% CI {ci[0] * 100:.0f}–{ci[1] * 100:.0f}%"


def _layout(fig: go.Figure, height: int = 320, **kw) -> go.Figure:
    title = kw.pop("title", None)
    legend = kw.get("showlegend", True)
    if title:  # the title sits above the legend row, never on top of it
        title = dict(title, x=0, xanchor="left", y=1, yanchor="top", yref="container", pad=dict(t=6, l=4))
    fig.update_layout(height=height + (26 if legend else 0), margin=dict(l=8, r=16, t=64 if legend else 36, b=8),
                      legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0, traceorder="normal"), title=title,
                      bargap=0.35, hoverlabel=dict(namelength=-1), **kw)
    return fig


def _rate_bars(names: list[str], values: list[float | None], cis: list[tuple], title: str, color: str) -> go.Figure:
    """One measure, one bar per run, with its confidence interval and a direct label."""
    vals = [None if v is None else v * 100 for v in values]
    err_hi = [0 if v is None or c[1] is None else c[1] * 100 - v for v, c in zip(vals, cis)]
    err_lo = [0 if v is None or c[0] is None else v - c[0] * 100 for v, c in zip(vals, cis)]
    fig = go.Figure(go.Bar(
        y=names, x=vals, orientation="h", marker_color=color, width=0.5,
        error_x=dict(type="data", symmetric=False, array=err_hi, arrayminus=err_lo, color=GRAY, thickness=1.2, width=3),
        hovertemplate="%{y}<br>" + title + " %{x:.1f}%<extra></extra>"))
    # values in their own column right of the plot, clear of the interval whiskers
    fig.add_scatter(y=names, x=[104] * len(names), mode="text", textposition="middle right", hoverinfo="skip",
                    text=["–" if v is None else f"{v:.1f}%" for v in vals], cliponaxis=False)
    fig.update_xaxes(range=[0, 128], ticksuffix="%", tickvals=[0, 25, 50, 75, 100])
    fig.update_yaxes(autorange="reversed")
    return _layout(fig, height=90 + 44 * len(names), title=dict(text=title, font=dict(size=14)), showlegend=False)


def _stacked_handlers(names: list[str], counts: list[dict[str, int]], title: str) -> go.Figure:
    """Who handled the requests, as shares of each row (100% stacked)."""
    fig = go.Figure()
    totals = [sum(c.values()) or 1 for c in counts]
    for h in L.HANDLERS:
        share = [c.get(h, 0) / t * 100 for c, t in zip(counts, totals)]
        fig.add_bar(y=names, x=share, name=L.HANDLER_LABELS[h], orientation="h", marker_color=HANDLER_COLORS[h],
                    marker_line=dict(width=2, color="rgba(255,255,255,0.9)"), width=0.5,
                    text=[f"{s:.0f}%" if s >= 8 else "" for s in share], textposition="inside", insidetextanchor="middle",
                    customdata=[c.get(h, 0) for c in counts],
                    hovertemplate="%{y}<br>" + L.HANDLER_LABELS[h] + "：%{customdata} 筆（%{x:.1f}%）<extra></extra>")
    fig.update_xaxes(range=[0, 100.5], ticksuffix="%", tickvals=[0, 25, 50, 75, 100])
    fig.update_yaxes(autorange="reversed")
    return _layout(fig, height=90 + 44 * len(names), barmode="stack", title=dict(text=title, font=dict(size=14)))


def _ecdf(series: dict[str, pd.Series], title: str) -> go.Figure:
    """Latency distributions as cumulative curves: read any percentile off the y axis."""
    fig = go.Figure()
    for i, (name, s) in enumerate(series.items()):
        s = pd.to_numeric(s, errors="coerce").dropna().sort_values()
        if s.empty:
            continue
        y = [(k + 1) / len(s) * 100 for k in range(len(s))]
        fig.add_scatter(x=s, y=y, name=name, mode="lines", line=dict(width=2, shape="hv", color=RUN_COLORS[i % 3],
                                                                   dash=["solid", "dash", "dot"][(i // 3) % 3]),
                        hovertemplate=name + "<br>%{y:.0f}% 的請求 ≤ %{x:,.0f} ms<extra></extra>")
    fig.update_xaxes(title="latency (ms)", rangemode="tozero")
    fig.update_yaxes(title="累積比例", ticksuffix="%", range=[0, 101])
    return _layout(fig, height=340, title=dict(text=title, font=dict(size=14)))


def _sample_table(df: pd.DataFrame, cols: list[str], key: str, height: int = 320) -> pd.Series | None:
    """A table of samples; selecting a row returns it for the detail panel."""
    cols = [c for c in cols if c in df]
    view = df[cols].reset_index(drop=True)
    cfg = {"prompt": st.column_config.TextColumn("prompt", width="large"),
           "response": st.column_config.TextColumn("response", width="large")}
    ev = st.dataframe(view, hide_index=True, height=min(height, 38 * (len(view) + 1) + 4), column_config=cfg, on_select="rerun",
                      selection_mode="single-row", key=key, width="stretch")
    sel = ev.selection.rows if ev and ev.selection else []
    st.download_button("下載 CSV", view.to_csv(index=False).encode("utf-8-sig"), f"{key}.csv", "text/csv", key=f"{key}_dl")
    return df.reset_index(drop=True).iloc[sel[0]] if sel else None


def _detail(row: pd.Series | None) -> None:
    if row is None:
        st.caption("點選表格中的一列，可以看完整內容。")
        return
    with st.container(border=True):
        st.markdown("**Prompt**")
        st.text(row.get("prompt") or "")
        if isinstance(row.get("response"), str) and row.get("response"):
            st.markdown("**模型回覆**")
            st.text(row["response"])
        facts = {k: row.get(k) for k in ("sample_id", "expected", "action", "blocked_reason", "texts", "refusal_label",
                                         "judge", "judge_reason", "categories", "error", "latency_ms", "gen_latency_ms")
                 if k in row and row.get(k) not in (None, "", []) and not (isinstance(row.get(k), float) and pd.isna(row.get(k)))}
        st.json(facts, expanded=True)
        if isinstance(row.get("raw"), str) and row.get("raw"):
            with st.expander("原始回覆（raw）"):
                st.code(row["raw"])


# --------------------------------------------------------------------------- pages
def page_overview(runs: list[L.Run]) -> None:
    st.header("總覽")
    table = pd.DataFrame([r.summary() for r in runs])
    kinds = sorted(table["類型"].unique())
    pick = st.segmented_control("類型", kinds, default=kinds, selection_mode="multi", key="ov_kind") or kinds
    table = table[table["類型"].isin(pick)]
    st.caption("recall：Guardrail 為「該擋的有擋下」的比例，模型為「有害請求被拒絕」的比例。"
               "FPR：正常請求被誤擋／被過度拒絕的比例。Latency 為呼叫端量到的時間（模型只計產生回覆，不含判斷模型）。")
    show = table.drop(columns=["run_id", "dataset_id", "run"]).copy()
    for c in ("recall", "FPR", "precision", "F1", "錯誤率"):
        show[c] = show[c] * 100
    pct = st.column_config.NumberColumn(format="%.1f%%")
    st.dataframe(show, hide_index=True, width="stretch",
                 column_config={"recall": pct, "FPR": pct, "precision": pct, "F1": pct, "錯誤率": pct,
                                "p50 ms": st.column_config.NumberColumn(format="%.0f"),
                                "p95 ms": st.column_config.NumberColumn(format="%.0f")})
    if len({d for d in table["dataset_id"]}) > 1:
        st.info("表中的 run 使用了不同的資料集，數字不能直接互相比較；請看「資料集」欄，或到比較頁挑同一資料集的 run。")
    sel = [r for r in runs if r.run_id in set(table["run_id"])]
    sums = [L.summarize(r.rows, r.task) for r in sel]
    c1, c2, c3 = st.columns(3)
    names = [r.label for r in sel]
    c1.plotly_chart(_rate_bars(names, [s.get("recall") for s in sums], [s.get("recall_ci", (None, None)) for s in sums],
                               "recall（越高越好）", BLUE), width="stretch", config=PLOT_CFG)
    c2.plotly_chart(_rate_bars(names, [s.get("fpr") for s in sums], [s.get("fpr_ci", (None, None)) for s in sums],
                               "FPR（越低越好）", ORANGE), width="stretch", config=PLOT_CFG)
    lat = go.Figure(go.Bar(y=names, x=[s.get("latency_p95") for s in sums], orientation="h", marker_color=AQUA, width=0.5,
                           text=[_ms(s.get("latency_p95")) for s in sums], textposition="outside", cliponaxis=False,
                           hovertemplate="%{y}<br>p95 %{x:,.0f} ms<extra></extra>"))
    lat.update_yaxes(autorange="reversed")
    lat.update_xaxes(rangemode="tozero", range=[0, max([s.get("latency_p95") or 0 for s in sums] + [1]) * 1.3])
    c3.plotly_chart(_layout(lat, height=90 + 44 * len(names), title=dict(text="p95 latency（越低越好）", font=dict(size=14)),
                            showlegend=False), width="stretch", config=PLOT_CFG)
    st.caption("誤差線為 95% 信賴區間（Wilson）。樣本少時區間很寬，差異落在區間內就不能下結論。")


def page_run(runs: list[L.Run]) -> None:
    st.header("Run 詳情")
    run = st.selectbox("Run", runs, format_func=lambda r: r.label, key="run_pick")
    df, task = run.rows, run.task
    labels = L.OUTCOME_LABELS.get(task, L.OUTCOME_LABELS["guardrail"])
    if df.empty:
        st.warning("這個 run 還沒有任何結果。")
        return
    s = L.summarize(df, task)
    sub = run.subject
    st.caption(" · ".join(str(x) for x in [
        {"guardrail": "Guardrail", "model_safety": "模型（bare）" if sub.get("mode") == "bare" else "模型"}.get(task, task),
        f"adapter {sub['adapter']}" if sub.get("adapter") else None,
        f"方向 {run.input_type}" if task == "guardrail" else None,
        f"LiteLLM {sub['litellm_version']}" if sub.get("litellm_version") else None,
        f"判斷模型 {run.info['judge'].get('type')} / {run.info['judge'].get('model')}" if run.info.get("judge") else None,
        f"資料集 {(run.info.get('dataset') or {}).get('name')}", f"狀態 {run.info.get('status')}"] if x))
    for w in run.info.get("warnings") or []:
        st.warning(w)
    if sub.get("service_changed_during_run"):
        st.warning("Guardrail service 的資訊在評測前後不一致：這個 run 可能混了兩種設定。")
    if run.service_info:
        with st.expander("Guardrail service 資訊（評測開始時由 service 回報）"):
            st.json(run.service_info, expanded=2)
    if task == "model_safety" and "blocked" in df and df["blocked"].fillna(False).any():
        nb = int(df["blocked"].fillna(False).sum())
        how = {"error": "未計入拒絕率（記為錯誤）", "refusal": "計為拒絕", "compliance": "計為未拒絕"}.get(
            (run.metrics.get("model_safety") or {}).get("on_blocked"), "")
        st.info(f"{nb} 筆請求在到達模型前被 LiteLLM 上的 guardrail 擋下（HTTP 400）；{how}。")
    k = st.columns(7)
    k[0].metric("樣本數", s["n"], help=f"該擋 {s['n_unsafe']}／不該擋 {s['n_safe']}")
    k[1].metric("recall", _pct(s["recall"]), help=_ci(s["recall_ci"]))
    k[2].metric("FPR", _pct(s["fpr"]), help=_ci(s["fpr_ci"]))
    k[3].metric("precision", _pct(s["precision"]))
    k[4].metric("錯誤率", _pct(s["error_rate"]), help="連線失敗或協定錯誤，不計入指標")
    k[5].metric("latency p50", _ms(s["latency_p50"]))
    k[6].metric("latency p95", _ms(s["latency_p95"]))

    left, right = st.columns([1, 1])
    with left:
        col = "action" if task == "guardrail" else "refusal_label"
        st.markdown(f"**expected × {'action' if task == 'guardrail' else '判斷結果'}**")
        st.dataframe(pd.crosstab(df["expected"], df[col].fillna("error"), margins=True, margins_name="合計"),
                     width="stretch")
        if task == "guardrail" and "texts_changed" in df and (df["action"] == "GUARDRAIL_INTERVENED").any():
            iv = df[df["action"] == "GUARDRAIL_INTERVENED"]
            st.caption(f"GUARDRAIL_INTERVENED {len(iv)} 筆，其中 {int(iv['texts_changed'].fillna(False).sum())} 筆的文字確實被改寫。")
    with right:
        by = st.segmented_control("切片", ["source", "language", "expected"], default="source", key="run_slice") or "source"
        sl = L.slice_table(df, by, task)
        for c in ("recall", "fpr"):
            sl[c] = sl[c] * 100
        pct = st.column_config.NumberColumn(format="%.1f%%")
        st.dataframe(sl, hide_index=True, width="stretch",
                     column_config={"recall": pct, "fpr": pct, "p50 ms": st.column_config.NumberColumn(format="%.0f"),
                                    "p95 ms": st.column_config.NumberColumn(format="%.0f")})

    st.subheader("樣本")
    f = st.columns([2, 1, 1, 2])
    default = [o for o in ("FN", "FP", "error") if (df["outcome"] == o).any()] or list(L.OUTCOMES)
    outs = f[0].multiselect("結果", list(L.OUTCOMES), default=default, format_func=lambda o: f"{labels[o]}（{o}）", key="run_out")
    srcs = f[1].multiselect("source", sorted(df["source"].dropna().unique()), key="run_src")
    langs = f[2].multiselect("language", sorted(df["language"].dropna().unique()), key="run_lang")
    kw = f[3].text_input("關鍵字（prompt／回覆）", key="run_kw")
    view = df[df["outcome"].isin(outs or L.OUTCOMES)]
    if srcs:
        view = view[view["source"].isin(srcs)]
    if langs:
        view = view[view["language"].isin(langs)]
    if kw:
        hay = view["prompt"].fillna("") + "\n" + view["response"].fillna("").astype(str)
        view = view[hay.str.contains(kw, case=False, regex=False)]
    view = view.assign(結果=view["outcome"].map(labels))
    st.caption(f"{len(view)} / {len(df)} 筆")
    lat_col = L.latency_column(task)
    cols = ["結果", "expected", "action" if task == "guardrail" else "refusal_label", "source", "language", "prompt",
            "response", lat_col, "error"]
    _detail(_sample_table(view, cols, "run_samples"))


def page_compare(runs: list[L.Run]) -> None:
    st.header("Run 比較")
    st.caption("比較同一類型的 run，例如不同的 guardrail，或不同的模型。只計算所有被選 run 都有結果的樣本。")
    kinds = {"Guardrail（request）": ("guardrail", "request"), "Guardrail（response）": ("guardrail", "response"),
             "模型": ("model_safety", "request")}
    avail = [k for k, (t, d) in kinds.items() if sum(r.task == t and r.input_type == d for r in runs) >= 1]
    kind = st.segmented_control("類型", avail, default=avail[0], key="cmp_kind") or avail[0]
    task, direction = kinds[kind]
    pool = [r for r in runs if r.task == task and r.input_type == direction]
    chosen = st.multiselect("Run", pool, default=pool[:4], format_func=lambda r: r.label, key=f"cmp_runs_{kind}")
    if len(chosen) < 2:
        st.info("請至少選兩個 run。")
        return
    aligned, info = L.align(chosen)
    if not info.get("same_dataset"):
        st.warning("這些 run 的資料集不同；以下只比較共同的樣本。")
    if aligned.empty:
        st.error("這些 run 沒有共同的樣本。")
        return
    st.caption(f"共同樣本 {info['common']} 筆")
    gold = aligned["gold_positive"].astype(bool)
    names = [r.label for r in chosen]
    rows, recs, fprs = [], [], []
    for r in chosen:
        p = aligned[r.label]
        tp, fn, fp, tn = int((gold & p).sum()), int((gold & ~p).sum()), int((~gold & p).sum()), int((~gold & ~p).sum())
        lat = L.latency_stats(aligned[f"{r.label}__ms"])
        recs.append((tp / (tp + fn) if tp + fn else None, L.wilson(tp, tp + fn)))
        fprs.append((fp / (fp + tn) if fp + tn else None, L.wilson(fp, fp + tn)))
        rows.append({"run": r.label, "recall": recs[-1][0], "漏擋": fn, "FPR": fprs[-1][0], "誤擋": fp,
                     "p50 ms": lat["p50"], "p95 ms": lat["p95"], "平均 ms": lat["mean"]})
    t = pd.DataFrame(rows)
    for c in ("recall", "FPR"):
        t[c] = t[c] * 100
    pct = st.column_config.NumberColumn(format="%.1f%%")
    num = st.column_config.NumberColumn(format="%.0f")
    st.dataframe(t, hide_index=True, width="stretch",
                 column_config={"recall": pct, "FPR": pct, "p50 ms": num, "p95 ms": num, "平均 ms": num})
    c1, c2 = st.columns(2)
    c1.plotly_chart(_rate_bars(names, [x[0] for x in recs], [x[1] for x in recs], "recall（越高越好）", BLUE),
                    width="stretch", config=PLOT_CFG)
    c2.plotly_chart(_rate_bars(names, [x[0] for x in fprs], [x[1] for x in fprs], "FPR（越低越好）", ORANGE),
                    width="stretch", config=PLOT_CFG)
    st.plotly_chart(_ecdf({r.label: aligned[f"{r.label}__ms"] for r in chosen}, "Latency 累積分佈"),
                    width="stretch", config=PLOT_CFG)

    if any(r.service_info for r in chosen):
        st.subheader("Guardrail service 資訊")
        diff_only = st.toggle("只顯示不同的設定", value=True, key="cmp_info_diff")
        info = L.info_table(chosen, diff_only)
        if info.empty:
            st.caption("這些 run 回報的 service 資訊完全相同。" if diff_only else "沒有 service 資訊。")
        else:
            st.dataframe(info, hide_index=True, width="stretch")

    st.subheader("兩個 run 的差異樣本")
    a_col, b_col = st.columns(2)
    a = a_col.selectbox("A", names, index=0, key="cmp_a")
    b = b_col.selectbox("B", [n for n in names if n != a], index=0, key="cmp_b")
    groups = L.disagreements(aligned, a, b)
    agree = int((aligned[a] == aligned[b]).sum())
    st.caption(f"判定一致 {agree} / {len(aligned)} 筆（{agree / len(aligned) * 100:.0f}%）")
    first = next((i for i, k in enumerate(groups) if len(groups[k])), 0)  # open on a group that has samples
    g = st.radio("差異類型", list(groups), index=first, format_func=lambda k: f"{k}：{len(groups[k])} 筆",
                 key=f"cmp_group_{a}_{b}")
    _detail(_sample_table(groups[g], ["expected", "source", "language", "prompt", "response", a, b], "cmp_samples", 260))


def page_combo(runs: list[L.Run]) -> None:
    st.header("模型 × Guardrail")
    st.caption("把 guardrail（request 方向）放在模型前面會怎樣：用兩份 run 的逐筆結果離線合併。"
               "被 guardrail 擋下的請求不會送到模型，所以 latency 只計 guardrail；其餘為 guardrail + 模型。")
    models = [r for r in runs if r.task == "model_safety"]
    guards = [r for r in runs if r.task == "guardrail" and r.input_type == "request"]
    if not models or not guards:
        st.info("需要至少一個模型 run 和一個 request 方向的 guardrail run。")
        return
    model = st.selectbox("模型", models, format_func=lambda r: r.label, key="combo_model")
    same = [g for g in guards if g.dataset_id == model.dataset_id]
    chosen = st.multiselect("Guardrail", guards, default=same or guards, format_func=lambda r: r.label, key="combo_guards")
    variants: list[tuple[str, L.Run | None]] = [("無 guardrail（模型本身）", None)] + [(g.label, g) for g in chosen]
    combos = {name: L.combine(model, g) for name, g in variants}
    if chosen:  # same samples for every variant, or the bars are not comparable
        common = set.intersection(*(set(d["sample_id"]) for d in combos.values()))
        combos = {n: d[d["sample_id"].isin(common)] for n, d in combos.items()}
    sums = {n: L.combo_summary(d) for n, d in combos.items()}
    base = sums["無 guardrail（模型本身）"]
    if not base.get("n"):
        st.error("模型 run 與所選 guardrail run 沒有共同的樣本。")
        return
    st.caption(f"共同樣本 {base['n']} 筆（有害 {base['n_unsafe']}／正常 {base['n_safe']}）")

    rows = []
    for n, s in sums.items():
        rows.append({
            "組合": n, "殘餘風險": s["residual_rate"], "風險降低": None if n == variants[0][0] or s["residual_rate"] is None
            else base["residual_rate"] - s["residual_rate"],
            "正常請求可用率": s["served_rate"], "可用率變化": None if n == variants[0][0] or s["served_rate"] is None
            else s["served_rate"] - base["served_rate"],
            "p50 ms": s["latency_p50"], "p95 ms": s["latency_p95"],
            "p95 增加 ms": None if n == variants[0][0] else s["latency_p95"] - base["latency_p95"]})
    t = pd.DataFrame(rows)
    for c in ("殘餘風險", "風險降低", "正常請求可用率", "可用率變化"):
        t[c] = t[c] * 100
    pct = st.column_config.NumberColumn(format="%.1f%%")
    pp = st.column_config.NumberColumn(format="%+.1f pp")
    num = st.column_config.NumberColumn(format="%.0f")
    st.dataframe(t, hide_index=True, width="stretch",
                 column_config={"殘餘風險": st.column_config.NumberColumn(format="%.1f%%", help="有害請求最後仍得到模型回答的比例"),
                                "風險降低": st.column_config.NumberColumn(format="%.1f pp", help="相對於模型本身，殘餘風險下降的百分點（收益）"),
                                "正常請求可用率": pct, "可用率變化": pp, "p50 ms": num, "p95 ms": num,
                                "p95 增加 ms": st.column_config.NumberColumn(format="%+.0f")})
    names = list(sums)
    c1, c2 = st.columns(2)
    c1.plotly_chart(_stacked_handlers(names, [sums[n]["unsafe"] for n in names], "有害請求：誰處理的（橘色 = 殘餘風險）"),
                    width="stretch", config=PLOT_CFG)
    c2.plotly_chart(_stacked_handlers(names, [sums[n]["safe"] for n in names], "正常請求：誰處理的（橘色 = 正常服務）"),
                    width="stretch", config=PLOT_CFG)
    st.plotly_chart(_ecdf({n: d["latency_ms"] for n, d in combos.items()}, "端到端 latency 累積分佈（guardrail + 模型）"),
                    width="stretch", config=PLOT_CFG)
    st.caption("Latency 是由兩份 run 各自量到的時間相加而得，不含 LiteLLM 呼叫 guardrail 的額外開銷；正式數字請以實際串接後的量測為準。")

    if not chosen:
        return
    st.subheader("樣本")
    c1, c2 = st.columns(2)
    pick = c1.selectbox("組合", names[1:], key="combo_pick")
    cats = {"殘餘風險：有害請求，guardrail 沒擋、模型也回答了": lambda d: d["gold_positive"].astype(bool) & (d["handler"] == "answered"),
            "Guardrail 補上的：模型會回答，但被 guardrail 擋下": lambda d: d["gold_positive"].astype(bool) & d["guard_blocked"] & ~d["model_refused"],
            "重複保護：guardrail 擋下，模型本來也會拒絕": lambda d: d["gold_positive"].astype(bool) & d["guard_blocked"] & d["model_refused"],
            "只靠模型：guardrail 沒擋，模型拒絕": lambda d: d["gold_positive"].astype(bool) & (d["handler"] == "model"),
            "代價：正常請求被 guardrail 誤擋": lambda d: ~d["gold_positive"].astype(bool) & d["guard_blocked"]}
    d = combos[pick]
    cat = c2.selectbox("類別", list(cats), format_func=lambda k: f"{k}（{int(cats[k](d).sum())} 筆）", key="combo_cat")
    _detail(_sample_table(d[cats[cat](d)], ["expected", "source", "language", "prompt", "response", "guard_action",
                                            "guard_ms", "model_ms"], "combo_samples", 260))


def page_latency(runs: list[L.Run]) -> None:
    st.header("Latency")
    st.caption("呼叫端量到的單筆請求時間。模型 run 只計產生回覆的時間，不含判斷模型。")
    chosen = st.multiselect("Run", runs, default=runs[:6], format_func=lambda r: r.label, key="lat_runs")
    if not chosen:
        return
    rows = []
    for r in chosen:
        col = L.latency_column(r.task)
        ok = r.rows[r.rows["outcome"] != "error"]
        rows.append({"run": r.label, "n": len(ok), **{k: v for k, v in L.latency_stats(ok[col]).items()}})
    num = st.column_config.NumberColumn(format="%.0f")
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                 column_config={k: num for k in ("p50", "p90", "p95", "p99", "mean", "max")})
    st.plotly_chart(_ecdf({r.label: r.rows.loc[r.rows["outcome"] != "error", L.latency_column(r.task)] for r in chosen},
                          "Latency 累積分佈"), width="stretch", config=PLOT_CFG)
    st.subheader("依判定結果")
    run = st.selectbox("Run", chosen, format_func=lambda r: r.label, key="lat_run")
    col = L.latency_column(run.task)
    df = run.rows[run.rows["outcome"] != "error"]
    key = "action" if run.task == "guardrail" else "refusal_label"
    out = [{key: v, "n": len(g), **L.latency_stats(g[col])} for v, g in df.groupby(key, dropna=False)]
    st.dataframe(pd.DataFrame(out), hide_index=True, width="stretch",
                 column_config={k: num for k in ("p50", "p90", "p95", "p99", "mean", "max")})
    df = df.assign(字數=df["prompt"].fillna("").str.len())
    fig = go.Figure(go.Scatter(x=df["字數"], y=df[col], mode="markers", marker=dict(color=BLUE, size=8, opacity=0.7),
                               customdata=df["prompt"].fillna("").str.slice(0, 80),
                               hovertemplate="%{customdata}<br>%{x} 字 · %{y:,.0f} ms<extra></extra>"))
    fig.update_xaxes(title="prompt 字數", rangemode="tozero")
    fig.update_yaxes(title="latency (ms)", rangemode="tozero")
    st.plotly_chart(_layout(fig, height=320, title=dict(text="Latency 與 prompt 長度", font=dict(size=14)), showlegend=False),
                    width="stretch", config=PLOT_CFG)


PAGES = {"總覽": page_overview, "Run 詳情": page_run, "Run 比較": page_compare, "模型 × Guardrail": page_combo,
         "Latency": page_latency}


def main() -> None:
    st.set_page_config(page_title="Guardrail 評測分析", layout="wide")
    with st.sidebar:
        st.title("Guardrail 評測分析")
        results_dir = st.text_input("results 目錄", _results_dir())
        page = st.radio("頁面", list(PAGES), key="page")
        if st.button("重新讀取"):
            st.cache_resource.clear()
    runs = _load(results_dir, L.results_fingerprint(results_dir))
    skipped = [r for r in runs if not r.supported]
    runs = [r for r in runs if r.supported]
    with st.sidebar:
        st.caption(f"{len(runs)} 個 run" + (f"；{len(skipped)} 個因 schema 版本不符而略過" if skipped else ""))
    if not runs:
        st.info(f"在 `{results_dir}` 找不到任何 run（需要含 run.json 的目錄）。先跑一次 `python -m src.evaluation ...`。")
        return
    PAGES[page](runs)


main()
