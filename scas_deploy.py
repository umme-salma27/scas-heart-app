"""SCAS deployment module (research prototype, not for clinical use without local validation).

Exact stacked prediction and explanation on the log-odds scale, "marginalise" handling of
not-measured inputs with a risk range, novelty warning, local recalibration and a
robustness badge computed over the Rashomon set of equally accurate stacks.
"""
import time
from math import factorial

import joblib
import numpy as np
import pandas as pd

F = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope"]
EPS = 1e-3
MIN_BG = 15
RANGE_POOL = 100
CHOL_MMOL_TO_MG = 38.67
SCHEMA = {
    "age": dict(label="Age", unit="years", kind="num", lo=18, hi=100),
    "sex": dict(label="Sex", kind="cat", codes={0: "female", 1: "male"}),
    "cp": dict(label="Chest pain type", kind="cat",
               codes={1: "typical angina", 2: "atypical angina", 3: "non-anginal pain", 4: "asymptomatic"}),
    "trestbps": dict(label="Resting blood pressure", unit="mm Hg", kind="num", lo=70, hi=250),
    "chol": dict(label="Serum cholesterol", unit="mg/dL", kind="num", lo=80, hi=700),
    "fbs": dict(label="Fasting blood sugar > 120 mg/dL", kind="cat", codes={0: "no", 1: "yes"}),
    "restecg": dict(label="Resting ECG", kind="cat",
                    codes={0: "normal", 1: "ST-T abnormality", 2: "left ventricular hypertrophy"}),
    "thalach": dict(label="Maximum heart rate (exercise test)", unit="bpm", kind="num", lo=60, hi=220),
    "exang": dict(label="Exercise-induced angina", kind="cat", codes={0: "no", 1: "yes"}),
    "oldpeak": dict(label="ST depression, exercise vs rest", unit="mm", kind="num", lo=-3, hi=7),
    "slope": dict(label="Slope of peak exercise ST segment", kind="cat",
                  codes={1: "upsloping", 2: "flat", 3: "downsloping"}),
}

_STRUCT = {}


def _struct(d):
    if d not in _STRUCT:
        ar = np.arange(2 ** d)
        M = ((ar[:, None] >> np.arange(d)) & 1).astype(bool)
        pc = M.sum(1)
        js = []
        for j in range(d):
            i0 = ar[((ar >> j) & 1) == 0]
            w = np.array([factorial(k) * factorial(d - k - 1) / factorial(d) for k in pc[i0]])
            js.append((i0, i0 | (1 << j), w))
        _STRUCT[d] = (M, js)
    return _STRUCT[d]


def sigmoid(z):
    return 1 / (1 + np.exp(-np.asarray(z, float)))


def learner_logit(model, X):
    clf = model[-1]
    name = type(clf).__name__
    if name in ("LogisticRegression", "ExplainableBoostingClassifier"):
        return np.asarray(model.decision_function(X), float)
    if name == "XGBClassifier":
        Xt = X if isinstance(model[0], str) else model[0].transform(X)
        return np.asarray(clf.predict(Xt, output_margin=True), float)
    p = np.clip(model.predict_proba(X)[:, 1], EPS, 1 - EPS)
    return np.log(p / (1 - p))


def encode_novelty(nov, x):
    val = {f: x[F.index(f)] for f in F}
    num = np.array([(nov["med"][f] if np.isnan(val[f]) else val[f]) for f in nov["num"]])
    parts = [(num - np.array([nov["mu"][f] for f in nov["num"]])) / np.array([nov["sd"][f] for f in nov["num"]]),
             np.array([(nov["mode"][f] if np.isnan(val[f]) else val[f]) for f in nov["bin"]])]
    parts += [np.array([float(val[f] == v) for v in vals]) for f, vals in nov["nom"].items()]
    parts.append(np.array([float(np.isnan(val[f])) for f in nov["miss_cols"]]))
    return np.concatenate(parts)


class SCASModel:
    def __init__(self, bundle):
        self.bundle = bundle
        self.names = list(bundle["model_names"])
        self.models = [bundle["models"][n] for n in self.names]
        self.G = np.asarray(bundle["background"], float)
        self.pool = np.asarray(bundle["pool"], float)
        self.arms = {k: (np.asarray(v["w"], float), float(v["b"])) for k, v in bundle["arms"].items()}
        self.rs_w = np.asarray(bundle["rashomon_w"], float)
        self.rs_b = np.asarray(bundle["rashomon_b"], float)
        self.nov = bundle["novelty"]
        self.default_arm = bundle["default_arm"]
        self.recal = {k: (0.0, 1.0) for k in self.arms}

    @classmethod
    def load(cls, path):
        return cls(joblib.load(path))

    def learner_logits(self, X):
        X = pd.DataFrame(np.asarray(X, float).reshape(-1, len(F)), columns=F)
        return np.column_stack([learner_logit(m, X) for m in self.models])

    def validate(self, record, chol_unit="mg/dL"):
        x, notes, errors = np.full(len(F), np.nan), [], []
        for j, f in enumerate(F):
            v = record.get(f)
            if v is None or (isinstance(v, str) and v.strip() == ""):
                continue
            try:
                v = float(v)
            except (TypeError, ValueError):
                errors.append(f"{SCHEMA[f]['label']}: not a number")
                continue
            if np.isnan(v):
                continue
            s = SCHEMA[f]
            if f == "chol":
                if v == 0:
                    notes.append("Cholesterol = 0 is a missing-value code in the source data; treated as not measured.")
                    continue
                if str(chol_unit).lower().startswith("mmol"):
                    v *= CHOL_MMOL_TO_MG
            if f == "slope" and v == 0:
                notes.append("ST slope = 0 treated as not measured.")
                continue
            if f == "trestbps" and v == 0:
                errors.append("Resting blood pressure = 0 is impossible; leave it empty if not measured.")
                continue
            if s["kind"] == "cat":
                if v != int(v) or int(v) not in s["codes"]:
                    errors.append(f"{s['label']}: allowed codes {list(s['codes'])}")
                    continue
            elif not s["lo"] <= v <= s["hi"]:
                errors.append(f"{s['label']} = {v:g} is outside the plausible range {s['lo']}-{s['hi']} {s.get('unit', '')}")
                continue
            x[j] = v
        return x, notes, errors

    def background_for(self, absent):
        if not absent.any():
            return self.G
        ok = ~np.isnan(self.G[:, absent]).any(1)
        if ok.sum() >= MIN_BG:
            return self.G[ok]
        okp = ~np.isnan(self.pool[:, absent]).any(1)
        return self.pool[okp][:len(self.G)]

    def learner_shap(self, x, absent):
        """Exact interventional Shapley values per learner; absent variables take background values."""
        Gm = self.background_for(absent)
        blocks, meta = [], []
        for bg in Gm:
            xx = x.copy()
            xx[absent] = bg[absent]
            D = np.where(~((xx == bg) | (np.isnan(xx) & np.isnan(bg))))[0]
            M, _ = _struct(len(D))
            H = np.repeat(bg[None, :], len(M), axis=0)
            if len(D):
                H[:, D] = np.where(M, xx[D], bg[D])
            blocks.append(H)
            meta.append((D, len(M)))
        V = self.learner_logits(np.vstack(blocks))
        phi = np.zeros((len(F), V.shape[1]))
        base, top, pos = np.zeros(V.shape[1]), np.zeros(V.shape[1]), 0
        for D, nM in meta:
            v = V[pos:pos + nM]
            pos += nM
            for j, (i0, i1, w) in enumerate(_struct(len(D))[1]):
                phi[D[j]] += w @ (v[i1] - v[i0])
            base += v[0]
            top += v[-1]
        return phi / len(Gm), base / len(Gm), top / len(Gm), len(Gm)

    def marginal_learner_logits(self, x, absent):
        if not absent.any():
            return self.learner_logits(x[None, :])[0]
        Gm = self.background_for(absent)
        H = np.repeat(x[None, :], len(Gm), axis=0)
        H[:, absent] = Gm[:, absent]
        return self.learner_logits(H).mean(0)

    def risk_range(self, x, absent, arm):
        if not absent.any():
            return None
        okp = ~np.isnan(self.pool[:, absent]).any(1)
        rows = self.pool[okp][:RANGE_POOL]
        H = np.repeat(x[None, :], len(rows), axis=0)
        H[:, absent] = rows[:, absent]
        w, b = self.arms[arm]
        a, beta = self.recal[arm]
        p = sigmoid(a + beta * (b + self.learner_logits(H) @ w))
        return float(np.quantile(p, 0.10)), float(np.quantile(p, 0.90))

    def robustness(self, phi_m, zbar, arm):
        w, b = self.arms[arm]
        a, beta = self.recal[arm]
        phi = phi_m @ w
        S = self.rs_w @ phi_m.T
        top = int(np.argmax(np.abs(phi)))
        agree = float((np.abs(S).argmax(1) == top).mean())
        sign = {F[j]: float((np.sign(S[:, j]) == np.sign(phi[j])).mean()) for j in range(len(F)) if abs(phi[j]) > 0.05}
        p_rs = sigmoid(a + beta * (self.rs_b + self.rs_w @ zbar))
        badge = "high" if agree >= 0.8 else ("moderate" if agree >= 0.5 else "low")
        return {"top_reason": F[top], "top_reason_agreement": agree, "badge": badge, "sign_agreement": sign,
                "rashomon_prob_range": (float(p_rs.min()), float(p_rs.max())), "n_stacks": len(self.rs_w)}

    def novelty(self, x):
        d = np.sqrt(((np.asarray(self.nov["ref"]) - encode_novelty(self.nov, x)) ** 2).sum(1))
        score = float(np.sort(d)[:self.nov["k"]].mean())
        return {"score": score, "threshold": float(self.nov["thr"]), "warning": bool(score > self.nov["thr"])}

    def assess(self, record, arm=None, mode="marginalise", chol_unit="mg/dL", explain=True):
        t0 = time.perf_counter()
        arm = arm or self.default_arm
        if isinstance(record, dict):
            x, notes, errors = self.validate(record, chol_unit)
        else:
            x, notes, errors = np.asarray(record, float).copy(), [], []
        if errors:
            return {"ok": False, "errors": errors, "notes": notes}
        missing = np.isnan(x)
        absent = missing if mode == "marginalise" else np.zeros(len(F), bool)
        w, b = self.arms[arm]
        a, beta = self.recal[arm]
        out = {"ok": True, "arm": arm, "mode": mode, "notes": notes,
               "not_measured": [F[j] for j in np.where(missing)[0]]}
        if explain:
            phi_m, base_m, zbar, n_bg = self.learner_shap(x, absent)
        else:
            zbar = self.marginal_learner_logits(x, absent)
        lp_raw = b + zbar @ w
        lp = a + beta * lp_raw
        out.update({"lp_raw": float(lp_raw), "lp": float(lp), "prob": float(sigmoid(lp)),
                    "risk_range": self.risk_range(x, absent, arm), "novelty": self.novelty(x),
                    "recalibrated": (a, beta) != (0.0, 1.0)})
        if explain:
            phi = beta * (phi_m @ w)
            base = a + beta * (b + base_m @ w)
            order = np.argsort(-np.abs(phi))
            out.update({"base_lp": float(base), "base_prob": float(sigmoid(base)), "n_background": n_bg,
                        "explanation": [{"feature": F[j], "label": SCHEMA[F[j]]["label"], "value": x[j],
                                         "not_measured": bool(missing[j]), "phi": float(phi[j])} for j in order],
                        "efficiency_error": float(abs(phi.sum() + base - lp)),
                        "robustness": self.robustness(phi_m, zbar, arm), "_phi_learners": phi_m})
        out["latency_ms"] = 1000 * (time.perf_counter() - t0)
        return out

    def fit_local_recalibration(self, records, y, arm=None, slope=False, mode="marginalise"):
        arm = arm or self.default_arm
        lp = np.array([self.assess(r, arm, mode, explain=False)["lp_raw"] for r in records])
        y = np.asarray(y, float)
        a, bt, pa, pb = 0.0, 1.0, 1 / 4, 1.0
        for _ in range(50):
            p = sigmoid(a + bt * lp)
            r, wt = p - y, p * (1 - p)
            if slope:
                g = np.array([r.sum() + pa * a, (r * lp).sum() + pb * (bt - 1)])
                H = np.array([[wt.sum() + pa, (wt * lp).sum()], [(wt * lp).sum(), (wt * lp * lp).sum() + pb]])
                st = np.linalg.solve(H, g)
                a, bt = a - st[0], bt - st[1]
            else:
                a -= (r.sum() + pa * a) / (wt.sum() + pa)
        p = sigmoid(a + bt * lp)
        se = float(1 / np.sqrt((p * (1 - p)).sum() + pa))
        return {"arm": arm, "a": float(a), "beta": float(bt), "se_a": se, "n": len(y),
                "recommended": bool(abs(a) > 2 * se), "observed_%": 100 * y.mean(),
                "predicted_before_%": 100 * sigmoid(lp).mean()}

    def set_recalibration(self, arm, a=0.0, beta=1.0):
        self.recal[arm] = (float(a), float(beta))
