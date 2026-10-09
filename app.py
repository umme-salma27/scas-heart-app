"""SCAS coronary artery disease risk app (research prototype, not for clinical use without local validation)."""
import copy
import html
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from guidelines import chest_pain_conflict, esc2024_category, guideline_context
from scas_deploy import F, SCHEMA, SCASModel, sigmoid

HERE = os.path.dirname(os.path.abspath(__file__))
ARM_LABELS = {"SCAS-C": "SCAS-C (clinically concordant, default)",
              "C1-Equal": "C1 (equal weights)",
              "SCAS-S": "SCAS-S (most stable across hospitals)"}
REQUIRED = ["age", "sex", "cp"]
EXERCISE = ["thalach", "exang", "oldpeak", "slope"]
LAYOUT = [[("Patient", "person", ["age", "sex"]), ("Symptoms", "stethoscope", ["cp"])],
          [("Resting tests", "monitor_heart", ["trestbps", "chol", "fbs", "restecg"])],
          [("Exercise test", "directions_run", EXERCISE)]]
STEP = {"age": 1.0, "trestbps": 1.0, "chol": 1.0, "thalach": 1.0, "oldpeak": 0.1}
FMT = {"age": "%.0f", "trestbps": "%.0f", "chol": "%.1f", "thalach": "%.0f", "oldpeak": "%.1f"}
MIN_LOCAL = 25
RED, RED_DARK, BLUE, GREY = "#C62828", "#8E1B1B", "#1565C0", "#9CA3AF"

st.set_page_config(page_title="SCAS cardiac risk (research prototype)", page_icon="🩺", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@24,500,1,0');
@import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.2/css/all.min.css');
html, body, .stApp, .stMarkdown, button, input, textarea, select {font-family: 'Inter', sans-serif;}
.block-container {padding-top: 1.4rem; padding-bottom: 2rem; max-width: 1280px;}
[data-testid="stDecoration"], footer {display: none;}
.msr {font-family: 'Material Symbols Rounded'; font-weight: normal; font-style: normal; line-height: 1;
      letter-spacing: normal; text-transform: none; white-space: nowrap; direction: ltr;
      -webkit-font-feature-settings: 'liga'; font-feature-settings: 'liga'; -webkit-font-smoothing: antialiased;}
.scas-header {background: linear-gradient(100deg, #8E1B1B 0%, #C62828 55%, #E53935 100%); color: #fff;
              border-radius: 16px; padding: 20px 26px; display: flex; align-items: center; gap: 20px;
              box-shadow: 0 6px 18px rgba(142, 27, 27, .18); margin-bottom: 14px;}
.scas-header .logo {background: #fff; border-radius: 50%; width: 68px; height: 68px; min-width: 68px;
                    display: flex; align-items: center; justify-content: center;}
.scas-header .logo i {font-size: 36px; color: #C62828;}
.scas-header .deco {margin-left: auto; display: flex; gap: 14px; opacity: .28;}
.scas-header .deco .msr {font-size: 64px; color: #fff;}
.scas-header h1 {color: #fff !important; font-size: 1.65rem; font-weight: 700; margin: 0; padding: 0;}
.scas-header .sub {margin-top: 4px; font-size: .96rem; opacity: .95;}
.scas-header .tags {margin-top: 8px; display: flex; gap: 8px; flex-wrap: wrap;}
.scas-header .tag {background: rgba(255,255,255,.18); border: 1px solid rgba(255,255,255,.35); border-radius: 999px;
                   padding: 2px 10px; font-size: .78rem;}
.scas-note {background: #FFF5F5; border-left: 4px solid #C62828; padding: 10px 14px; border-radius: 8px;
            color: #5F2120; font-size: .86rem; margin-bottom: 12px;}
.sec {display: flex; align-items: center; gap: 8px; font-weight: 650; font-size: 1.05rem; color: #1F2937;
      margin: 2px 0 6px;}
.sec .msr {color: #C62828; font-size: 22px; background: #FFEBEE; border-radius: 8px; padding: 4px;}
.card {background: #fff; border: 1px solid #F3D6D6; border-radius: 14px; padding: 16px 18px; height: 100%;
       box-shadow: 0 1px 6px rgba(17, 24, 39, .05);}
.card .label {color: #6B7280; font-size: .76rem; text-transform: uppercase; letter-spacing: .06em; font-weight: 600;}
.card .big {font-size: 3.1rem; font-weight: 700; color: #C62828; line-height: 1.05; margin-top: 4px;}
.card .mid {font-size: 1.25rem; font-weight: 650; color: #1F2937; margin-top: 6px;}
.card .small {color: #4B5563; font-size: .86rem; margin-top: 6px;}
.riskbar {height: 12px; background: linear-gradient(90deg, #FDECEA 0%, #F28B82 50%, #B71C1C 100%);
          border-radius: 7px; position: relative; margin: 14px 0 4px;}
.riskbar .range {position: absolute; top: 0; height: 12px; background: rgba(31, 41, 55, .28); border-radius: 7px;}
.riskbar .marker {position: absolute; top: -5px; width: 4px; height: 22px; background: #111827; border-radius: 2px;
                  transform: translateX(-2px);}
.riskscale {display: flex; justify-content: space-between; color: #9CA3AF; font-size: .72rem;}
.chip {display: inline-flex; align-items: center; gap: 5px; padding: 4px 12px; border-radius: 999px;
       font-weight: 650; font-size: .9rem; margin-top: 6px;}
.chip .msr {font-size: 18px;}
.chip-high {background: #E8F5E9; color: #1B5E20;}
.chip-moderate {background: #FFF8E1; color: #8D6E00;}
.chip-low, .chip-warn {background: #FFEBEE; color: #B71C1C;}
.chip-ok {background: #E8F5E9; color: #1B5E20;}
.reason {display: flex; gap: 10px; align-items: flex-start; padding: 8px 0; border-bottom: 1px dashed #F3D6D6;}
.reason:last-child {border-bottom: none;}
.reason .msr {font-size: 22px;}
.up {color: #C62828;} .down {color: #1565C0;}
.gtab {width: 100%; border-collapse: collapse; font-size: .9rem; margin-top: 4px;}
.gtab td, .gtab th {padding: 7px 8px; border-bottom: 1px solid #F3D6D6; text-align: left; vertical-align: top;}
.gtab th {color: #6B7280; font-size: .74rem; text-transform: uppercase; letter-spacing: .05em; font-weight: 600;}
.gtab td.num {font-weight: 650; color: #1F2937; white-space: nowrap;}
.gwarn {background: #FFF8E1; border-left: 4px solid #F9A825; padding: 9px 13px; border-radius: 8px; color: #5D4400;
        font-size: .86rem; margin-top: 10px;}
.foot {color: #9CA3AF; font-size: .78rem; text-align: center; margin-top: 26px; border-top: 1px solid #F3D6D6;
       padding-top: 10px;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    m = SCASModel.load(os.path.join(HERE, "scas_bundle.joblib"))
    for mod in m.models:
        if "n_jobs" in mod[-1].get_params():
            mod[-1].set_params(n_jobs=1)
    return m


@st.cache_data
def load_text(name):
    path = os.path.join(HERE, name)
    return open(path, encoding="utf-8").read() if os.path.exists(path) else ""


def session_model():
    base = load_model()
    if "recal" not in st.session_state:
        st.session_state.recal = {k: (0.0, 1.0) for k in base.arms}
    m = copy.copy(base)
    m.recal = st.session_state.recal
    return m


def icon(name):
    return f'<span class="msr">{name}</span>'


def section(title, ic):
    st.markdown(f'<div class="sec">{icon(ic)}{html.escape(title)}</div>', unsafe_allow_html=True)


def show_value(f, v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "not measured"
    s = SCHEMA[f]
    if s["kind"] == "cat":
        return s["codes"].get(int(v), str(v))
    return f"{v:g} {s.get('unit', '')}".strip()


def set_inputs(rec, unit="mg/dL"):
    for f in F:
        v = rec.get(f)
        if v is not None and SCHEMA[f]["kind"] == "cat":
            v = int(v)
        elif v is not None:
            v = float(v)
        st.session_state[f"in_{f}"] = v
    st.session_state.chol_unit = unit
    st.session_state.no_ex = False


def current_record():
    rec = {f: st.session_state.get(f"in_{f}") for f in F}
    if st.session_state.get("no_ex"):
        for f in EXERCISE:
            rec[f] = None
    return rec


def waterfall(out, title):
    items = [e for e in out["explanation"] if not e["not_measured"]]
    n = len(items)
    fig, ax = plt.subplots(figsize=(7.6, 0.46 * n + 1.9), dpi=150)
    cum = out["base_lp"]
    rows = n + 1
    ax.barh(rows, cum, color="#E5E7EB", height=0.55)
    ax.text(cum, rows, f"  {100 * sigmoid(cum):.0f}%", va="center", fontsize=8.5, color="#374151")
    labels = [f"Baseline (average patient)"]
    for k, e in enumerate(items):
        y = n - k
        left = cum if e["phi"] >= 0 else cum + e["phi"]
        ax.barh(y, abs(e["phi"]), left=left, color=RED if e["phi"] > 0 else BLUE, height=0.55)
        ax.plot([cum, cum], [y + 0.72, y - 0.28], color="#9CA3AF", lw=0.6, ls=":")
        cum += e["phi"]
        ax.text(max(cum, cum - e["phi"]), y, f"  {e['phi']:+.2f}", va="center", fontsize=8.5,
                color=RED_DARK if e["phi"] > 0 else BLUE)
        labels.append(f"{SCHEMA[e['feature']]['label']} = {show_value(e['feature'], e['value'])}")
    ax.barh(0, cum, color=RED_DARK, height=0.55)
    ax.text(cum, 0, f"  {100 * sigmoid(cum):.0f}%", va="center", fontsize=9, fontweight="bold", color=RED_DARK)
    labels.append("This patient")
    ax.set_yticks([rows] + [n - k for k in range(n)] + [0])
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.axvline(0, color="#6B7280", lw=0.7)
    lo, hi = ax.get_xlim()
    ax.set_xlim(lo - 0.15, hi + 0.45)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color("#D1D5DB"); ax.spines["bottom"].set_color("#D1D5DB")
    ax.tick_params(colors="#374151", labelsize=8.5)
    ax.grid(axis="x", color="#F3F4F6", lw=0.8); ax.set_axisbelow(True)
    ax.set_xlabel("Log-odds of coronary artery disease (exact Shapley contributions)", fontsize=8.5, color="#374151")
    sec = ax.secondary_xaxis("top", functions=(lambda z: z, lambda z: z))
    ticks = [p for p in [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95] if lo - 0.15 <= np.log(p / (1 - p)) <= hi + 0.45]
    sec.set_xticks([np.log(p / (1 - p)) for p in ticks])
    sec.set_xticklabels([f"{100 * p:.0f}%" for p in ticks], fontsize=8, color="#6B7280")
    sec.spines["top"].set_color("#D1D5DB")
    ax.set_title(title, fontsize=10, color="#111827", loc="left", pad=22)
    fig.tight_layout()
    return fig


def reasons_html(out, k=3):
    rows = []
    for e in [e for e in out["explanation"] if not e["not_measured"] and abs(e["phi"]) >= 0.05][:k]:
        up = e["phi"] > 0
        cls = "up" if up else "down"
        rows.append(f'<div class="reason"><span class="msr {cls}">{"arrow_upward" if up else "arrow_downward"}</span>'
                    f'<div><b>{html.escape(SCHEMA[e["feature"]]["label"])}</b> = '
                    f'{html.escape(show_value(e["feature"], e["value"]))}<br>'
                    f'<span class="{"up" if up else "down"}">{"raises" if up else "lowers"} the risk</span> '
                    f'<span style="color:#6B7280">({e["phi"]:+.2f} log-odds)</span></div></div>')
    return "".join(rows) or '<div class="small">No single variable moves the risk by more than 0.05 log-odds.</div>'


def guideline_html(out, rec):
    age, sex, cp = float(rec["age"]), int(rec["sex"]), int(rec["cp"])
    g = guideline_context(age, sex, cp)
    p = out["prob"]

    def pct(v):
        return "-" if v is None else f"{100 * v:.0f}%"

    def cat(v):
        return "-" if v is None else esc2024_category(v)

    clamp = " (nearest age band)"
    esc_note = ("Not defined for patients without chest pain." if g["esc2019"] is None else
                "Age, sex and chest pain only." + (clamp if g["esc2019_age_clamped"] else ""))
    df_note = "Age, sex and chest pain only; era-matched to the training data." + (clamp if g["df1979_age_clamped"] else "")
    rows = [("SCAS model (this result)", pct(p), cat(p), "Uses resting and exercise-test results as well."),
            ("ESC 2019 pretest probability", pct(g["esc2019"]), cat(g["esc2019"]), esc_note),
            ("Diamond-Forrester 1979", pct(g["df1979"]), cat(g["df1979"]), df_note)]
    body = "".join(f'<tr><td>{html.escape(a)}</td><td class="num">{b}</td><td>{html.escape(c)}</td>'
                   f'<td style="color:#6B7280">{html.escape(d)}</td></tr>' for a, b, c, d in rows)
    s = (f'<table class="gtab"><tr><th>Source</th><th>Probability</th><th>ESC 2024 likelihood category</th>'
         f'<th>Note</th></tr>{body}</table>'
         '<div class="small" style="margin-top:8px">The model is not a pretest-probability tool: it uses test results, '
         'so differences from the guideline values show the information added by those results. In symptomatic '
         'patients of new hospitals the model discriminated better than ESC 2019 pretest probability (AUC 0.731 vs '
         '0.583) but over-predicted (+6.2 points), so local recalibration is advised.</div>')
    phi_cp = next(e["phi"] for e in out["explanation"] if e["feature"] == "cp")
    if chest_pain_conflict(cp, phi_cp):
        s += ('<div class="gwarn"><b>Chest pain contribution differs from the guidelines.</b> Here, '
              f'{html.escape(SCHEMA["cp"]["codes"][cp])} {"lowers" if phi_cp < 0 else "raises"} the estimated risk '
              f'({phi_cp:+.2f} log-odds). In the 1981-1988 referral cohorts used for training, patients without chest '
              'pain had more CAD (79.0%) than patients with typical angina (43.5%), the opposite of all pretest tables, '
              'most likely because of referral bias. The model reproduces the data; weigh the guideline pretest '
              'probability above when interpreting this patient.</div>')
    return s


def json_ready(out):
    clean = {k: v for k, v in out.items() if not k.startswith("_")}
    return json.loads(json.dumps(clean, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))


def local_logits(model, df, mode, unit):
    Z, y, skipped = [], [], 0
    for r in df.to_dict("records"):
        rec = {f: (None if pd.isna(r.get(f)) else r.get(f)) for f in F}
        x, _, errors = model.validate(rec, unit)
        out = r.get("outcome")
        if errors or out is None or pd.isna(out) or float(out) not in (0.0, 1.0):
            skipped += 1
            continue
        absent = np.isnan(x) if mode == "marginalise" else np.zeros(len(F), bool)
        Z.append(model.marginal_learner_logits(x, absent))
        y.append(float(out))
    return np.array(Z), np.array(y), skipped


def fit_intercept(lp, y, prior_precision=0.25):
    a = 0.0
    for _ in range(50):
        p = sigmoid(a + lp)
        a -= ((p - y).sum() + prior_precision * a) / ((p * (1 - p)).sum() + prior_precision)
    p = sigmoid(a + lp)
    return float(a), float(1 / np.sqrt((p * (1 - p)).sum() + prior_precision))


model = session_model()
examples = json.loads(load_text("examples.json") or "{}")

# ---------------- sidebar ----------------
with st.sidebar:
    st.markdown(f'<div class="sec">{icon("tune")}Settings</div>', unsafe_allow_html=True)
    arms = list(model.arms)
    arm = st.selectbox("Stack weights", arms, index=arms.index(model.default_arm),
                       format_func=lambda a: ARM_LABELS.get(a, a), key="arm")
    compare = st.checkbox("Compare all three weightings side by side", key="compare")
    with st.expander("Advanced (research)", icon=":material/science:"):
        mode = st.radio("Values that were not measured", ["marginalise", "indicator"], key="mode",
                        format_func=lambda m: {"marginalise": "Average over measured patients (recommended)",
                                               "indicator": "Missing-value indicator (as trained; can encode hospital practice)"}[m])
    with st.expander("Local recalibration", icon=":material/local_hospital:"):
        st.write("Upload patients from your own hospital with the verified outcome (1 = CAD, 0 = no CAD). "
                 "Only the intercept is adjusted, and only when the local offset is larger than twice its "
                 f"standard error. About 50 patients are enough; fewer than {MIN_LOCAL} are not used.")
        template = pd.DataFrame([{**{f: None for f in F}, "outcome": None}])
        st.download_button("Download CSV template", template.to_csv(index=False), "local_patients_template.csv",
                           "text/csv", icon=":material/download:")
        unit_local = st.radio("Cholesterol unit in the file", ["mg/dL", "mmol/L"], horizontal=True, key="unit_local")
        up = st.file_uploader("CSV with the 11 variables + outcome", type="csv", key="recal_file")
        if up is not None:
            st.session_state.recal_df = pd.read_csv(up)
        force = st.checkbox("Apply even if not statistically needed", key="recal_force")
        c1, c2 = st.columns(2)
        fit_clicked = c1.button("Fit", key="recal_fit", disabled="recal_df" not in st.session_state)
        if c2.button("Reset", key="recal_reset"):
            st.session_state.recal = {k: (0.0, 1.0) for k in model.arms}
            st.session_state.pop("recal_table", None)
            model.recal = st.session_state.recal
        if fit_clicked:
            df_loc = st.session_state.recal_df
            missing_cols = [c for c in F + ["outcome"] if c not in df_loc.columns]
            if missing_cols:
                st.error(f"Missing columns: {missing_cols}")
            else:
                with st.spinner("Fitting local intercepts..."):
                    Z, y, skipped = local_logits(model, df_loc, mode, unit_local)
                if len(y) < MIN_LOCAL:
                    st.error(f"Only {len(y)} usable patients (skipped {skipped}); at least {MIN_LOCAL} are needed.")
                else:
                    rows = []
                    for k, (w, b) in model.arms.items():
                        lp = b + Z @ w
                        a, se = fit_intercept(lp, y)
                        rec_ok = abs(a) > 2 * se
                        applied = rec_ok or force
                        st.session_state.recal[k] = (a, 1.0) if applied else (0.0, 1.0)
                        rows.append({"weights": k, "patients": len(y), "observed_%": 100 * y.mean(),
                                     "predicted_before_%": 100 * sigmoid(lp).mean(),
                                     "predicted_after_%": 100 * sigmoid(lp + (a if applied else 0.0)).mean(),
                                     "offset_a": a, "SE": se, "needed (|a| > 2 SE)": rec_ok, "applied": applied})
                    st.session_state.recal_table = pd.DataFrame(rows)
                    st.session_state.recal_skipped = skipped
                    model.recal = st.session_state.recal
        if "recal_table" in st.session_state:
            st.dataframe(st.session_state.recal_table.round(3), hide_index=True)
            if st.session_state.get("recal_skipped"):
                st.caption(f"{st.session_state.recal_skipped} rows skipped (invalid values or outcome).")
    with st.expander("Model card", icon=":material/description:"):
        st.markdown(load_text("model_card.md") or "Model card not found.")

# ---------------- header ----------------
st.markdown(f"""
<div class="scas-header">
  <div class="logo"><i class="fa-solid fa-user-doctor"></i></div>
  <div>
    <h1>SCAS Cardiac Risk Assistant</h1>
    <div class="sub">Probability of coronary artery disease with exact, source-consistent explanations</div>
    <div class="tags"><span class="tag">Exact Shapley explanation</span><span class="tag">Missing-value aware</span>
    <span class="tag">Robustness badge</span><span class="tag">Novelty warning</span><span class="tag">Local recalibration</span>
    <span class="tag">Guideline context</span></div>
  </div>
  <div class="deco">{icon("ecg_heart")}{icon("stethoscope")}</div>
</div>
<div class="scas-note"><b>Research prototype.</b> Estimates the probability of angiographic coronary artery disease
(&gt; 50% narrowing in at least one major vessel) in adults referred for evaluation of suspected CAD. Not a diagnosis,
not a screening tool and not a 10-year risk score. Trained on 918 patients from four hospitals (1981-1988); validate
and recalibrate locally before clinical use. Outputs are information for qualified clinicians, not treatment advice;
testing and treatment decisions follow current guidelines and clinical judgement. Entered data are not stored.</div>
""", unsafe_allow_html=True)

if examples:
    cols = st.columns([1, 1, 1.35, 0.7])
    for c, (name, ex) in zip(cols, examples.items()):
        c.button(f"Example: {name}", key=f"ex_{name}", on_click=set_inputs, args=(ex["record"],),
                 icon=":material/person:", width="stretch")
    cols[-1].button("Clear", key="ex_clear", on_click=set_inputs, args=({},), icon=":material/restart_alt:",
                    width="stretch")

# ---------------- inputs ----------------
no_ex = st.session_state.get("no_ex", False)
for gc, groups in zip(st.columns(len(LAYOUT)), LAYOUT):
    for gname, gicon, feats in groups:
        with gc, st.container(border=True):
            section(gname, gicon)
            if gname == "Exercise test":
                st.checkbox("Exercise test not done", key="no_ex")
            for f in feats:
                s = SCHEMA[f]
                req = f in REQUIRED
                ph = "required" if req else "not measured"
                disabled = f in EXERCISE and no_ex
                label = s["label"] + (" *" if req else "")
                if s["kind"] == "cat":
                    st.selectbox(label, list(s["codes"]), format_func=lambda c, s=s: s["codes"][c], index=None,
                                 placeholder=ph, key=f"in_{f}", disabled=disabled)
                else:
                    unit = "unit below" if f == "chol" else s.get("unit", "")
                    st.number_input(f"{label} ({unit})", value=None, step=STEP[f], format=FMT[f], placeholder=ph,
                                    key=f"in_{f}", disabled=disabled)
                    if f == "chol":
                        st.radio("Cholesterol unit", ["mg/dL", "mmol/L"], horizontal=True, key="chol_unit",
                                 label_visibility="collapsed")

b1, b2 = st.columns([1, 3])
assess = b1.button("Assess patient", type="primary", key="assess", icon=":material/monitor_heart:", width="stretch")
b2.caption("Fields marked with an asterisk are required. Leave a field empty if it was not measured; the explanation "
           "then gives it zero weight and shows the range of risks over plausible values.")

if assess:
    rec = current_record()
    miss_req = [SCHEMA[f]["label"] for f in REQUIRED if rec[f] is None]
    if miss_req:
        st.session_state.result = {"ok": False, "errors": [f"Required: {', '.join(miss_req)}"], "notes": []}
    else:
        unit = st.session_state.get("chol_unit", "mg/dL")
        res = {"record": rec, "unit": unit, "arm": arm, "mode": mode,
               "main": model.assess(rec, arm=arm, mode=mode, chol_unit=unit)}
        if compare and res["main"]["ok"]:
            res["all"] = {k: (res["main"] if k == arm else model.assess(rec, arm=k, mode=mode, chol_unit=unit))
                          for k in model.arms}
        st.session_state.result = res

# ---------------- results ----------------
res = st.session_state.get("result")
if res is not None and not res.get("main", res).get("ok", False):
    for e in res.get("main", res).get("errors", []):
        st.error(e, icon=":material/error:")
elif res is not None:
    out = res["main"]
    if res["record"] != current_record() or res["arm"] != arm or res["mode"] != mode:
        st.warning("Inputs or settings changed since the last assessment; press **Assess patient** again.",
                   icon=":material/sync_problem:")
    st.markdown("")
    p = out["prob"]
    rr = out["risk_range"]
    rb = out["robustness"]
    nv = out["novelty"]
    range_html = (f'<div class="range" style="left:{100 * rr[0]:.1f}%; width:{100 * (rr[1] - rr[0]):.1f}%"></div>'
                  if rr else "")
    range_txt = (f"Range if the missing values were measured: <b>{100 * rr[0]:.0f}-{100 * rr[1]:.0f}%</b>"
                 if rr else "All inputs measured")
    badge_icon = {"high": "verified", "moderate": "info", "low": "warning"}[rb["badge"]]
    plo, phi_ = rb["rashomon_prob_range"]
    uncertain = plo < 0.5 < phi_
    nm = [SCHEMA[f]["label"] for f in out["not_measured"]]
    c1, c2, c3 = st.columns([1.15, 1, 1])
    c1.markdown(f"""<div class="card"><div class="label">Estimated probability of CAD</div>
        <div class="big">{100 * p:.0f}%</div>
        <div class="riskbar">{range_html}<div class="marker" style="left:{100 * p:.1f}%"></div></div>
        <div class="riskscale"><span>0%</span><span>25%</span><span>50%</span><span>75%</span><span>100%</span></div>
        <div class="small">{range_txt}<br>Baseline (average patient): {100 * out['base_prob']:.0f}%
        {'<br>Locally recalibrated' if out['recalibrated'] else ''}</div></div>""", unsafe_allow_html=True)
    c2.markdown(f"""<div class="card"><div class="label">Explanation robustness</div>
        <div class="chip chip-{rb['badge']}">{icon(badge_icon)}{rb['badge'].upper()}</div>
        <div class="mid">Main reason: {html.escape(SCHEMA[rb['top_reason']]['label'])}</div>
        <div class="small">{100 * rb['top_reason_agreement']:.0f}% of {rb['n_stacks']} equally accurate stacks agree
        on the main reason.{'' if rb['badge'] == 'high' else ' The ranking of reasons depends on the model choice.'}</div>
        <div class="small">{'Equally accurate models disagree on whether risk is above 50% (' + f'{100 * plo:.0f}-{100 * phi_:.0f}%).'
                            if uncertain else f'Equally accurate models: {100 * plo:.0f}-{100 * phi_:.0f}%.'}</div>
        </div>""", unsafe_allow_html=True)
    c3.markdown(f"""<div class="card"><div class="label">Input check</div>
        <div class="chip {'chip-warn' if nv['warning'] else 'chip-ok'}">{icon('warning' if nv['warning'] else 'check_circle')}
        {'Unusual patient' if nv['warning'] else 'Similar to training patients'}</div>
        <div class="small">Novelty score {nv['score']:.2f} (warning above {nv['threshold']:.2f}).</div>
        <div class="small"><b>Not measured:</b> {html.escape(', '.join(nm)) if nm else 'none'}</div>
        {''.join(f'<div class="small">{html.escape(n)}</div>' for n in out['notes'])}</div>""",
                unsafe_allow_html=True)

    st.markdown("")
    with st.container(border=True):
        section("Guideline context", "menu_book")
        st.markdown(guideline_html(out, res["record"]), unsafe_allow_html=True)

    st.markdown("")
    left, right = st.columns([1, 1.7])
    with left:
        with st.container(border=True):
            section("Main reasons", "clinical_notes")
            st.markdown(reasons_html(out), unsafe_allow_html=True)
            st.markdown('<div class="small" style="margin-top:10px">Red raises and blue lowers the risk. Contributions '
                        'are exact Shapley values of the deployed stack and add up from the baseline to this patient.'
                        '</div>', unsafe_allow_html=True)
        st.download_button("Download result (JSON)", json.dumps(json_ready(out), indent=2), "scas_result.json",
                           "application/json", icon=":material/download:", width="stretch")
    with right:
        with st.container(border=True):
            section("Explanation", "analytics")
            fig = waterfall(out, f"{ARM_LABELS.get(res['arm'], res['arm'])}")
            st.pyplot(fig)
            plt.close(fig)

    if "all" in res:
        with st.container(border=True):
            section("Comparison of the three weightings", "compare_arrows")
            comp = pd.DataFrame([{"weights": k, "risk (%)": round(100 * o["prob"], 1),
                                  "range if measured (%)": (f"{100 * o['risk_range'][0]:.0f}-{100 * o['risk_range'][1]:.0f}"
                                                            if o["risk_range"] else "-"),
                                  "main reason": SCHEMA[o["robustness"]["top_reason"]]["label"],
                                  "robustness": o["robustness"]["badge"]} for k, o in res["all"].items()])
            st.dataframe(comp, hide_index=True)
            contrib = pd.DataFrame({k: {SCHEMA[e["feature"]]["label"]: round(e["phi"], 3) for e in o["explanation"]}
                                    for k, o in res["all"].items()})
            contrib.index.name = "contribution (log-odds)"
            st.dataframe(contrib, height=35 * (len(contrib) + 1) + 3)

    with st.expander("Technical details", icon=":material/settings:"):
        w, b = model.arms[res["arm"]]
        st.write(f"Weights: {dict(zip(model.names, np.round(w, 3).tolist()))}; intercept {b:+.3f}; "
                 f"mode: {res['mode']}; background patients: {out['n_background']}")
        per = pd.DataFrame(out["_phi_learners"], index=[SCHEMA[f]["label"] for f in F], columns=model.names)
        per["stack (weighted)"] = per[model.names].values @ w
        st.dataframe(per.round(3))
        st.write(f"Efficiency check |sum of contributions + baseline - log-odds| = {out['efficiency_error']:.1e}; "
                 f"computed in {out['latency_ms']:.0f} ms.")

st.markdown('<div class="foot">SCAS - Source-Consistent Attribution Stacking &middot; research prototype &middot; '
            'exact explanations of a five-learner logit stack &middot; see the model card in the sidebar</div>',
            unsafe_allow_html=True)
