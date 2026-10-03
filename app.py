"""SCAS coronary artery disease risk app (research prototype, not for clinical use without local validation)."""
import copy
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from scas_deploy import F, SCHEMA, SCASModel, sigmoid

HERE = os.path.dirname(os.path.abspath(__file__))
ARM_LABELS = {"SCAS-C": "SCAS-C (clinically concordant, default)",
              "C1-Equal": "C1 (equal weights)",
              "SCAS-S": "SCAS-S (most stable across hospitals)"}
REQUIRED = ["age", "sex", "cp"]
EXERCISE = ["thalach", "exang", "oldpeak", "slope"]
GROUPS = [("Patient", ["age", "sex"]), ("Symptoms", ["cp"]),
          ("Resting tests", ["trestbps", "chol", "fbs", "restecg"]), ("Exercise test", EXERCISE)]
STEP = {"age": 1.0, "trestbps": 1.0, "chol": 1.0, "thalach": 1.0, "oldpeak": 0.1}
MIN_LOCAL = 25

st.set_page_config(page_title="SCAS heart-disease risk (research prototype)", layout="wide")


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


def contribution_chart(out, title):
    ex = out["explanation"][::-1]
    labels = [f"{SCHEMA[e['feature']]['label']} = {show_value(e['feature'], None if e['not_measured'] else e['value'])}"
              for e in ex]
    colors = ["lightgrey" if e["not_measured"] else ("tab:red" if e["phi"] > 0 else "tab:blue") for e in ex]
    fig, ax = plt.subplots(figsize=(7.5, 0.42 * len(ex) + 1.2))
    ax.barh(range(len(ex)), [e["phi"] for e in ex], color=colors)
    ax.set_yticks(range(len(ex)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.axvline(0, color="k", lw=0.6)
    ax.set_xlabel("Contribution to the log-odds of CAD (exact Shapley value)")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    return fig


def reasons_text(out, k=3):
    lines = []
    for e in [e for e in out["explanation"] if not e["not_measured"] and abs(e["phi"]) >= 0.05][:k]:
        verb = "raises" if e["phi"] > 0 else "lowers"
        lines.append(f"- **{SCHEMA[e['feature']]['label']}** = {show_value(e['feature'], e['value'])} "
                     f"**{verb}** the risk ({e['phi']:+.2f} log-odds)")
    return "\n".join(lines) if lines else "- No single variable moves the risk by more than 0.05 log-odds."


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
    st.header("Settings")
    arms = list(model.arms)
    arm = st.selectbox("Stack weights", arms, index=arms.index(model.default_arm),
                       format_func=lambda a: ARM_LABELS.get(a, a), key="arm")
    compare = st.checkbox("Compare all three weightings side by side", key="compare")
    with st.expander("Advanced (research)"):
        mode = st.radio("Values that were not measured", ["marginalise", "indicator"], key="mode",
                        format_func=lambda m: {"marginalise": "Average over measured patients (recommended)",
                                               "indicator": "Missing-value indicator (as trained; can encode hospital practice)"}[m])
    with st.expander("Local recalibration"):
        st.write("Upload patients from your own hospital with the verified outcome (1 = CAD, 0 = no CAD). "
                 "Only the intercept is adjusted, and only when the local offset is larger than twice its "
                 f"standard error. About 50 patients are enough; fewer than {MIN_LOCAL} are not used.")
        template = pd.DataFrame([{**{f: None for f in F}, "outcome": None}])
        st.download_button("Download CSV template", template.to_csv(index=False), "local_patients_template.csv",
                           "text/csv")
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
    with st.expander("Model card"):
        st.markdown(load_text("model_card.md") or "Model card not found.")

# ---------------- main page ----------------
st.title("SCAS: coronary artery disease risk with exact explanations")
st.info("**Research prototype.** Estimates the probability of angiographic coronary artery disease (> 50% narrowing "
        "in at least one major vessel) for adults referred for evaluation of suspected CAD. Not a diagnosis, not a "
        "screening tool and not a 10-year risk score. Trained on 918 patients from four hospitals (1981-1988); "
        "validate and recalibrate locally before any clinical use. Entered data are not stored.")

if examples:
    cols = st.columns(len(examples) + 1)
    for c, (name, ex) in zip(cols, examples.items()):
        c.button(f"Example: {name}", key=f"ex_{name}", on_click=set_inputs, args=(ex["record"],))
    cols[-1].button("Clear", key="ex_clear", on_click=set_inputs, args=({},))

no_ex = st.session_state.get("no_ex", False)
gcols = st.columns(len(GROUPS))
for gc, (gname, feats) in zip(gcols, GROUPS):
    with gc:
        st.subheader(gname)
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
                st.number_input(f"{label} ({unit})", value=None, step=STEP[f], placeholder=ph, key=f"in_{f}",
                                disabled=disabled)
                if f == "chol":
                    st.radio("Cholesterol unit", ["mg/dL", "mmol/L"], horizontal=True, key="chol_unit",
                             label_visibility="collapsed")

st.caption("Fields marked with an asterisk are required. Leave a field empty if it was not measured; the "
           "explanation then gives it zero weight and shows the range of risks over plausible values.")

if st.button("Assess", type="primary", key="assess"):
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

res = st.session_state.get("result")
if res is not None and not res.get("main", res).get("ok", False):
    for e in res.get("main", res).get("errors", []):
        st.error(e)
elif res is not None:
    out = res["main"]
    if res["record"] != current_record() or res["arm"] != arm or res["mode"] != mode:
        st.warning("Inputs or settings changed since the last assessment; press **Assess** again.")
    st.divider()
    left, right = st.columns([1, 1.4])
    with left:
        st.metric("Estimated probability of CAD", f"{100 * out['prob']:.0f}%")
        if out["risk_range"]:
            lo, hi = out["risk_range"]
            st.write(f"**Range if the missing values were measured:** {100 * lo:.0f}-{100 * hi:.0f}% "
                     f"(10th-90th percentile over measured training patients). A wide range means that measuring "
                     f"them could change the decision.")
        st.write(f"Average patient in the explanation background (baseline): {100 * out['base_prob']:.0f}%")
        rb = out["robustness"]
        msg = (f"**Explanation robustness: {rb['badge'].upper()}.** {100 * rb['top_reason_agreement']:.0f}% of "
               f"{rb['n_stacks']} equally accurate stacks give the same main reason "
               f"({SCHEMA[rb['top_reason']]['label']}).")
        {"high": st.success, "moderate": st.info, "low": st.warning}[rb["badge"]](
            msg + ("" if rb["badge"] == "high" else " The ranking of reasons depends on the model choice; "
                   "interpret it with caution."))
        plo, phi_ = rb["rashomon_prob_range"]
        if plo < 0.5 < phi_:
            st.warning(f"Equally accurate models disagree on whether the risk is above or below 50% "
                       f"(range {100 * plo:.0f}-{100 * phi_:.0f}%).")
        nv = out["novelty"]
        if nv["warning"]:
            st.warning("This patient is unusual compared with the training patients "
                       f"(novelty score {nv['score']:.2f} > {nv['threshold']:.2f}). Interpret with caution.")
        for n in out["notes"]:
            st.info(n)
        if out["recalibrated"]:
            a, b_ = model.recal[res["arm"]]
            st.caption(f"Locally recalibrated (intercept offset {a:+.2f}).")
        st.markdown("**Main reasons**")
        st.markdown(reasons_text(out))
        if out["not_measured"]:
            st.caption("Not measured (zero contribution): " + ", ".join(SCHEMA[f]["label"] for f in out["not_measured"]))
    with right:
        fig = contribution_chart(out, f"{ARM_LABELS.get(res['arm'], res['arm'])}: baseline "
                                      f"{100 * out['base_prob']:.0f}% -> this patient {100 * out['prob']:.0f}%")
        st.pyplot(fig)
        plt.close(fig)
        st.caption("Red bars raise the risk, blue bars lower it. The contributions are exact and add up from the "
                   "baseline log-odds to this patient's log-odds.")

    if "all" in res:
        st.subheader("Comparison of the three weightings")
        comp = pd.DataFrame([{"weights": k, "risk_%": round(100 * o["prob"], 1),
                              "range_%": (f"{100 * o['risk_range'][0]:.0f}-{100 * o['risk_range'][1]:.0f}"
                                          if o["risk_range"] else "-"),
                              "main reason": SCHEMA[o["robustness"]["top_reason"]]["label"],
                              "robustness": o["robustness"]["badge"]} for k, o in res["all"].items()])
        st.dataframe(comp, hide_index=True)
        contrib = pd.DataFrame({k: {SCHEMA[e["feature"]]["label"]: round(e["phi"], 3) for e in o["explanation"]}
                                for k, o in res["all"].items()})
        st.dataframe(contrib)

    st.download_button("Download result (JSON)", json.dumps(json_ready(out), indent=2), "scas_result.json",
                       "application/json")
    with st.expander("Technical details"):
        w, b = model.arms[res["arm"]]
        st.write(f"Weights: {dict(zip(model.names, np.round(w, 3).tolist()))}; intercept {b:+.3f}; "
                 f"mode: {res['mode']}; background patients: {out['n_background']}")
        per = pd.DataFrame(out["_phi_learners"], index=[SCHEMA[f]["label"] for f in F], columns=model.names)
        per["stack (weighted)"] = per[model.names].values @ w
        st.dataframe(per.round(3))
        st.write(f"Efficiency check |sum of contributions + baseline - log-odds| = {out['efficiency_error']:.1e}; "
                 f"computed in {out['latency_ms']:.0f} ms; novelty score {out['novelty']['score']:.2f} "
                 f"(threshold {out['novelty']['threshold']:.2f}).")
