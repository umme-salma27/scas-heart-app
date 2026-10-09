# Model card: SCAS coronary artery disease risk (research prototype)

## Intended use
Estimated probability of angiographic coronary artery disease (>50% narrowing in at least one major vessel) in adults referred for evaluation of suspected CAD. Research prototype; not a screening tool, not a 10-year risk score, not a diagnosis. Requires local validation.

Not intended for: screening of the general population, 10-year risk prediction, diagnosis, or use without local
validation and recalibration.

## Model
- Five base learners (logistic regression, random forest, extra trees, XGBoost, explainable boosting machine)
  combined by a simplex-weighted logit stack with a free intercept.
- Default weights: **SCAS-C**. All weightings in the app:

| Weights | Learner weights | Intercept |
|---|---|---|
| C1-Equal | LR 0.20, RF 0.20, ET 0.20, XGB 0.20, EBM 0.20 | +0.154 |
| SCAS-C | LR 0.50, RF 0.20, ET 0.10, XGB 0.00, EBM 0.20 | +0.195 |
| SCAS-S | LR 0.35, RF 0.12, ET 0.14, XGB 0.17, EBM 0.23 | +0.166 |

- SCAS-C is chosen by Source-Consistent Attribution Stacking: among stacks within 5% of the best unseen-hospital
  log-loss, it has the best combination of explanation stability and clinical concordance.
- Inputs: the 11 clinical variables shown in the app. Missing values are allowed except age, sex and chest pain type.

## Training data
918 unique patients from Cleveland, Hungary, Switzerland and VA Long Beach (1981-1988), cleaned. Duplicates removed; physiologically impossible zero codes and silently filled values in
the distributed benchmark were recoded as missing (the filled values were outcome-dependent and leak the label).

## Performance (leave-one-hospital-out, mean of 4 unseen hospitals)
- SCAS-C: AUC 0.817 (95% CI 0.760-0.867), log-loss 0.453, Brier 0.146; equal weights 0.817 / 0.449 / 0.143;
  CardiaTics stack 0.788 under the same protocol.
- Calibration does not transfer across hospitals (Switzerland under-predicted by about 22 points). Intercept-only
  recalibration on about 50 local patients removed this offset (-21.8 to -0.8 points; log-loss -44%).
- Clinical concordance of explanations (direction agreement with prior knowledge): SCAS-C 0.84 vs 0.79 for equal weights;
  the advantage held in all 50 random reference sets and for background sizes of 17, 33 and 64 patients.
- Nested hyperparameter tuning of the five learners did not change discrimination (AUC difference +0.0001,
  95% CI -0.007 to 0.006), so library defaults are used.

## Explanations
- Exact interventional Shapley values on the log-odds scale with a shared background of 33 patients (8 per hospital).
  Deployed values match offline exact values to 2.4e-15; contributions add up exactly
  (error 2.2e-15).
- Values that were not measured are treated as absent players: they receive zero contribution, the prediction
  averages over measured training patients, and a 10th-90th percentile risk range is shown.
- Robustness badge: share of 4090 simplex stacks within 1% of the best unseen-hospital log-loss that agree on the main reason (high >= 80%,
  moderate >= 50%, else low). On 160 reference patients, 91% received "high" and
  2% "low" with the default weights.
- Novelty warning: mean distance to 10 nearest training patients; warning above 95th percentile (leave-one-out).
- Median time per patient for a full assessment: about 0.5 s on 2 CPUs.

## Clinical interpretation (guideline context)
- The app shows the ESC 2019 pretest probability (Table 5; symptomatic patients only) and the Diamond-Forrester 1979
  pretest probability next to the model result, each with its ESC 2024 likelihood category (very low <= 5%,
  low > 5-15%, moderate > 15-50%, high > 50-85%, very high > 85%).
- The model uses test results and is not a pretest-probability tool. In symptomatic patients of unseen hospitals:
  AUC SCAS-C 0.731 vs ESC 2019 0.583, ACC/AHA 2002 0.547, Diamond-Forrester 0.560; the stacked models had the highest
  net benefit between about 10% and 55% risk. ESC 2019 under-predicted this cohort (-15.5 points); SCAS-C over-predicted
  it (+6.2 points).
- Explanations agree with 11 of 12 guideline statements (age, sex, typical vs non-anginal pain, resting blood
  pressure, cholesterol, diabetes, resting ECG, low maximum heart rate, exercise angina, ST depression, flat or
  downsloping ST segment). The exception is chest pain: in the training cohorts patients without chest pain had more
  CAD (79.0%) than patients with typical angina (43.5%), probably from referral bias. The app flags patients whose
  chest-pain contribution runs against the guidelines.
- Outputs are information, not treatment advice. Design follows the WHO guidance on ethics and governance of AI for
  health (2021): transparency (exact explanations), safety (intended use, novelty warning), inclusiveness (subgroup
  results below), accountability (this model card), privacy (no stored data), responsiveness (local recalibration).

## Limitations
- Four referral hospitals from 1981-1988 (USA, Hungary, Switzerland); mostly men (79%), middle-aged. Performance in
  women, in other countries (for example South Asia), in screening populations or with modern testing is unknown.
- Outcome is angiographic stenosis, not clinical events.
- Discrimination is similar in women and men (AUC 0.841 vs 0.847), but risk in women is over-predicted
  (+8.0 vs +0.3 points). Discrimination is lower at age >= 55 (0.805 vs 0.886).
- The chest-pain variable conflicts with guideline pretest tables (see above).
- Missingness of a variable carried hospital-specific meaning in the training data; the app therefore does not use
  missingness as a predictor by default.

## Ethics and privacy
No entered data are stored. Outputs are decision support for qualified clinicians in research settings only.

## Versions
Python 3.13.15; streamlit 1.65.0, numpy 2.1.3, pandas 2.2.3, scipy 1.16.3, scikit-learn 1.6.1, xgboost 3.4.1, interpret-core 0.7.8, joblib 1.6.0, matplotlib 3.10.0. Bundle created 2026-10-03 14:24.
