"""Guideline pretest probability and ESC 2024 likelihood categories (same tables as the paper, Cell 18).

sex: 1 = male, 0 = female; cp: 1 typical angina, 2 atypical angina, 3 non-anginal pain, 4 asymptomatic.
"""
import numpy as np

# Diamond & Forrester, N Engl J Med 1979;300:1350-1358; ages 30-39 / 40-49 / 50-59 / 60-69
DF_T = {1: {1: [.697, .873, .920, .943], 2: [.218, .461, .589, .671], 3: [.052, .141, .215, .281], 4: [.019, .055, .097, .123]},
        0: {1: [.258, .552, .794, .906], 2: [.042, .133, .324, .544], 3: [.008, .028, .084, .186], 4: [.003, .010, .032, .075]}}
DF_EDGES = [30, 40, 50, 60]
# ESC 2019 chronic coronary syndromes guideline (Eur Heart J 2020;41:407-477), Table 5 (chest pain columns);
# ages 30-39 / 40-49 / 50-59 / 60-69 / 70+; not defined for patients without symptoms
ESC_T = {1: {1: [.03, .22, .32, .44, .52], 2: [.04, .10, .17, .26, .34], 3: [.01, .03, .11, .22, .24]},
         0: {1: [.05, .10, .13, .16, .27], 2: [.03, .06, .06, .11, .19], 3: [.01, .02, .03, .06, .10]}}
ESC_EDGES = [30, 40, 50, 60, 70]

# ESC 2024 chronic coronary syndromes guideline (Eur Heart J 2024;45:3415-3537), likelihood of obstructive CAD
ESC24_EDGES = [0.05, 0.15, 0.50, 0.85]
ESC24_LABELS = ["very low (<=5%)", "low (>5-15%)", "moderate (>15-50%)", "high (>50-85%)", "very high (>85%)"]


def _lookup(table, edges, age, sex, cp):
    row = table.get(int(sex), {}).get(int(cp))
    if row is None:
        return None, False
    k = int(np.clip(np.searchsorted(edges, age, side="right") - 1, 0, len(edges) - 1))
    outside = age < edges[0] or (len(row) == 4 and age >= edges[-1] + 10)
    return float(row[k]), bool(outside)


def esc2024_category(p):
    return ESC24_LABELS[int(np.searchsorted(ESC24_EDGES, p, side="left"))]


def guideline_context(age, sex, cp):
    esc, esc_out = _lookup(ESC_T, ESC_EDGES, age, sex, cp)
    df, df_out = _lookup(DF_T, DF_EDGES, age, sex, cp)
    return {"esc2019": esc, "esc2019_age_clamped": esc_out, "df1979": df, "df1979_age_clamped": df_out}


def chest_pain_conflict(cp, phi_cp):
    """True when the chest-pain contribution runs against guideline ordering (training-data referral bias)."""
    return (int(cp) in (1, 2) and phi_cp < -0.05) or (int(cp) == 4 and phi_cp > 0.05)
