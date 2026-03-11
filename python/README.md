# Colorectal Cancer Trial Failure Prediction
## NLP + Machine Learning on ClinicalTrials.gov Protocol Text

![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-orange?logo=scikit-learn)
![Data](https://img.shields.io/badge/Data-ClinicalTrials.gov%20API%20v2-green)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

---

## Research Question

> **Can NLP applied to colorectal cancer clinical trial protocol text predict trial failure (early termination / low accrual) before recruitment begins — enabling upstream intervention?**

Roughly 40% of registered clinical trials terminate early, wasting resources, exposing participants to risk without benefit, and delaying evidence generation. If protocol text contains early signals of structural weaknesses — overly restrictive eligibility, ambiguous endpoints, unrealistic enrolment targets — a classifier trained on historical trial outcomes could flag high-risk protocols at registration.

---

## Methods

| Step | Method | Library |
|------|--------|---------|
| Data extraction | ClinicalTrials.gov API v2 (real data) | `requests` |
| Text preprocessing | Tokenisation, TF-IDF (1–2 gram, 5,000 features) | `scikit-learn` |
| Semantic embeddings | Latent Semantic Analysis via TruncatedSVD (BERT proxy) | `scikit-learn` |
| Classification | Logistic Regression, Random Forest, Gradient Boosting, SVM | `scikit-learn` |
| Calibration | Isotonic regression (CalibratedClassifierCV), Brier score | `scikit-learn` |
| Explainability | SHAP LinearExplainer — feature attribution per prediction | `shap` |
| Online learning | Prequential (test-then-train) prospective simulation | `scikit-learn` |
| Evaluation | ROC-AUC, Average Precision, Brier score, 5-fold stratified CV | `scikit-learn` |

### Why LSA instead of BERT?
Full BERT fine-tuning requires GPU. LSA via TruncatedSVD on TF-IDF produces dense semantic embeddings that serve as a CPU-safe approximation. The comparison (Cell 8) quantifies the AUC difference. Full transformer fine-tuning is described in the extension roadmap below.

---

## Results Summary

| Metric | Value |
|--------|-------|
| CRC trials analysed | _live from API_ |
| Overall failure rate | _computed at runtime_ |
| Best classifier | _highest test AUC at runtime_ |
| Top SHAP feature | _computed at runtime_ |
| TF-IDF vocabulary | 5,000 features |
| LSA components | 100 |

> All values are computed from live ClinicalTrials.gov data pulled at runtime. Re-running Cell 2 retrieves the current state of the registry.

---

## Figures

| Figure | Description |
|--------|-------------|
| `crc_fig1_data_overview.png` | Trial status distribution, failure rate by year, failure by sponsor class |
| `crc_fig2_tfidf_terms.png` | Top TF-IDF terms in terminated vs completed trials |
| `crc_fig3_roc_pr.png` | ROC curves and Precision-Recall curves for all 4 classifiers |
| `crc_fig4_shap.png` | SHAP feature importance — top 20 terms, direction of effect |
| `crc_fig5_calibration.png` | Reliability diagrams and Brier scores |
| `crc_fig6_online_learning.png` | Prequential AUC/AP curves and concept drift in failure rate |
| `crc_fig7_embeddings_learning.png` | TF-IDF vs LSA AUC comparison + learning curve |

---

## Repository Structure

```
colorectal-cancer-nlp-trial-prediction/
├── python/
│   ├── crc_nlp_trial_prediction.ipynb   # Main notebook (12 cells)
│   ├── crc_trials_raw.csv               # Extracted trial data
│   ├── crc_top_terms.csv                # TF-IDF term analysis
│   ├── crc_classifier_results.csv       # Model performance table
│   ├── crc_roc_curves.csv               # ROC data (all classifiers)
│   ├── crc_pr_curves.csv                # PR data (all classifiers)
│   ├── crc_shap_values.csv              # SHAP feature importance
│   ├── crc_embedding_comparison.csv     # TF-IDF vs LSA AUC
│   ├── crc_lsa_variance.csv             # LSA explained variance
│   ├── crc_online_learning.csv          # Prequential performance
│   ├── crc_calibration.csv              # Calibration data
│   └── crc_kpi.csv                      # Summary KPIs
├── requirements.txt
└── README.md
```

---

## Setup & Usage

```bash
# Clone
git clone https://github.com/sm-hasanulbari/colorectal-cancer-nlp-trial-prediction.git
cd colorectal-cancer-nlp-trial-prediction

# Install dependencies
pip install -r requirements.txt

# Run notebook
jupyter notebook python/crc_nlp_trial_prediction.ipynb
```

Run cells in order. Cell 2 fetches live data (~2 minutes). Cells 6–7 train classifiers (~3–5 minutes depending on hardware).

---

## Relevance to EDICT MSCA Doctoral Network

This project directly addresses the methodological core of **Project 6 (University of Galway)** — developing algorithmic systems to detect disparities in clinical trial recruitment:

| EDICT Project 6 Requirement | Demonstrated Here |
|------------------------------|-------------------|
| Programming proficiency in Python | Full pipeline: API → preprocessing → ML → visualisation |
| ML algorithm development | 4 classifiers, calibration, cross-validation |
| Explainable AI outputs | SHAP feature attribution with clinical interpretation |
| Online / adaptive learning | Prequential deployment simulation |
| GDPR-compliant data governance | Public registry data only; no personal data processed |
| Data pipeline design | API → text extraction → feature engineering → model outputs |
| Version control (Git/GitHub) | Full commit history, public repository |

---

## Extension Roadmap

- **Full BERT fine-tuning** on trial text using HuggingFace `transformers` (GPU required)
- **Multi-label classification**: predict specific failure reasons (safety, accrual, funding)
- **Active learning**: query-by-committee to minimise labelling burden
- **Site-level integration**: combine protocol text with site performance data
- **Prospective deployment**: RESTful API wrapper for real-time registry monitoring

---

## References

| Reference | Role in this analysis |
|-----------|-----------------------|
| Pedregosa et al. (2011) JMLR 12:2825–2830 | scikit-learn framework |
| Lundberg & Lee (2017) NeurIPS | SHAP explainability |
| Devlin et al. (2019) NAACL — BERT | Transformer embedding context |
| Deerwester et al. (1990) JASIST 41:391–407 | LSA / Truncated SVD |
| Gama et al. (2014) ACM Comput Surv 46(4):44 | Concept drift, online learning |
| Fogel (2018) Contemp Clin Trials Commun 11:156–164 | Clinical trial failure rates |
| Ghadessi et al. (2020) Orphanet J Rare Dis 15:184 | Predictors of trial termination |
| Saito & Rehmsmeier (2015) PLOS ONE 10(3):e0118432 | PR curves for imbalanced data |
| Brier (1950) Mon Weather Rev 78:1–3 | Proper scoring rules / calibration |

---

## Author

**Sm Hasan ul Bari** — MBBS · MSc Advanced Biostatistics & Epidemiology (Erasmus Mundus, Distinction) · MSc Health Economics & Decision Modelling (Sheffield, Merit)  
Associate HTA Analyst, NICE Manchester · Technical Consultant, UNICEF NYC  
[github.com/sm-hasanulbari](https://github.com/sm-hasanulbari) · [ORCID: 0000-0002-5209-2029](https://orcid.org/0000-0002-5209-2029)
