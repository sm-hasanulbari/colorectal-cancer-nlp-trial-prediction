# Colorectal Cancer Trial Failure Prediction

## NLP + Machine Learning on ClinicalTrials.gov Protocol Text

!\[Python](https://img.shields.io/badge/Python-3.13-blue?logo=python)
!\[scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-orange?logo=scikit-learn)
!\[Data](https://img.shields.io/badge/Data-ClinicalTrials.gov%20API%20v2-green)
!\[Status](https://img.shields.io/badge/Status-Complete-brightgreen)

\---

## Research Question

> \*\*Can NLP applied to colorectal cancer clinical trial protocol text predict trial failure (early termination / low accrual) before recruitment begins — enabling upstream intervention?\*\*

Roughly 40% of registered clinical trials terminate early, wasting resources, exposing participants to risk without benefit, and delaying evidence generation. If protocol text contains early signals of structural weaknesses — overly restrictive eligibility, ambiguous endpoints, unrealistic enrolment targets — a classifier trained on historical trial outcomes could flag high-risk protocols at registration.

\---

## Methods

|Step|Method|Library|
|-|-|-|
|Data extraction|ClinicalTrials.gov API v2 (real data)|`requests`|
|Text preprocessing|Tokenisation, TF-IDF (1–2 gram, 5,000 features)|`scikit-learn`|
|Semantic embeddings|Latent Semantic Analysis via TruncatedSVD (BERT proxy)|`scikit-learn`|
|Classification|Logistic Regression, Random Forest, Gradient Boosting, SVM|`scikit-learn`|
|Calibration|Isotonic regression (CalibratedClassifierCV), Brier score|`scikit-learn`|
|Explainability|SHAP LinearExplainer — feature attribution per prediction|`shap`|
|Online learning|Prequential (test-then-train) prospective simulation|`scikit-learn`|
|Evaluation|ROC-AUC, Average Precision, Brier score, 5-fold stratified CV|`scikit-learn`|

### Why LSA instead of BERT?

Full BERT fine-tuning requires GPU. LSA via TruncatedSVD on TF-IDF produces dense semantic embeddings that serve as a CPU-safe approximation. The comparison (Cell 8) quantifies the AUC difference. Full transformer fine-tuning is described in the extension roadmap below.

\---

## Results Summary

|Metric|Value|
|-|-|
|CRC trials analysed|*4927*|
|Overall failure rate|*15.6%*|
|Best classifier|*Logistic Regression*|
|Best test AUC|*0.713*|
|Top SHAP feature|tumor|

> All values are computed from live ClinicalTrials.gov data pulled at runtime. Re-running Cell 2 retrieves the current state of the registry.

\---

## Figures

|Figure|Description|
|-|-|
|`crc\_fig1\_data\_overview.png`|Trial status distribution, failure rate by year, failure by sponsor class|
|`crc\_fig2\_tfidf\_terms.png`|Top TF-IDF terms in terminated vs completed trials|
|`crc\_fig3\_roc\_pr.png`|ROC curves and Precision-Recall curves for all 4 classifiers|
|`crc\_fig4\_shap.png`|SHAP feature importance — top 20 terms, direction of effect|
|`crc\_fig5\_calibration.png`|Reliability diagrams and Brier scores|
|`crc\_fig6\_online\_learning.png`|Prequential AUC/AP curves and concept drift in failure rate|
|`crc\_fig7\_embeddings\_learning.png`|TF-IDF vs LSA AUC comparison + learning curve|

\---

## Repository Structure

```
colorectal-cancer-nlp-trial-prediction/
├── python/
│   ├── crc\_nlp\_trial\_prediction.ipynb   # Main notebook (12 cells)
│   ├── crc\_trials\_raw.csv               # Extracted trial data
│   ├── crc\_top\_terms.csv                # TF-IDF term analysis
│   ├── crc\_classifier\_results.csv       # Model performance table
│   ├── crc\_roc\_curves.csv               # ROC data (all classifiers)
│   ├── crc\_pr\_curves.csv                # PR data (all classifiers)
│   ├── crc\_shap\_values.csv              # SHAP feature importance
│   ├── crc\_embedding\_comparison.csv     # TF-IDF vs LSA AUC
│   ├── crc\_lsa\_variance.csv             # LSA explained variance
│   ├── crc\_online\_learning.csv          # Prequential performance
│   ├── crc\_calibration.csv              # Calibration data
│   └── crc\_kpi.csv                      # Summary KPIs
├── requirements.txt
└── README.md
```

\---

## Setup \& Usage

```bash
# Clone
git clone https://github.com/sm-hasanulbari/colorectal-cancer-nlp-trial-prediction.git
cd colorectal-cancer-nlp-trial-prediction

# Install dependencies
pip install -r requirements.txt

# Run notebook
jupyter notebook python/crc\_nlp\_trial\_prediction.ipynb
```

Run cells in order. Cell 2 fetches live data (\~2 minutes). Cells 6–7 train classifiers (\~3–5 minutes depending on hardware).

\---

## Relevance to EDICT MSCA Doctoral Network

This project directly addresses the methodological core of **Project 6 (University of Galway)** — developing algorithmic systems to detect disparities in clinical trial recruitment:

|EDICT Project 6 Requirement|Demonstrated Here|
|-|-|
|Programming proficiency in Python|Full pipeline: API → preprocessing → ML → visualisation|
|ML algorithm development|4 classifiers, calibration, cross-validation|
|Explainable AI outputs|SHAP feature attribution with clinical interpretation|
|Online / adaptive learning|Prequential deployment simulation|
|GDPR-compliant data governance|Public registry data only; no personal data processed|
|Data pipeline design|API → text extraction → feature engineering → model outputs|
|Version control (Git/GitHub)|Full commit history, public repository|

\---

## Extension Roadmap

* **Full BERT fine-tuning** on trial text using HuggingFace `transformers` (GPU required)
* **Multi-label classification**: predict specific failure reasons (safety, accrual, funding)
* **Active learning**: query-by-committee to minimise labelling burden
* **Site-level integration**: combine protocol text with site performance data
* **Prospective deployment**: RESTful API wrapper for real-time registry monitoring

\---

## References

|Reference|Role in this analysis|
|-|-|
|Pedregosa et al. (2011) JMLR 12:2825–2830|scikit-learn framework|
|Lundberg \& Lee (2017) NeurIPS|SHAP explainability|
|Devlin et al. (2019) NAACL — BERT|Transformer embedding context|
|Deerwester et al. (1990) JASIST 41:391–407|LSA / Truncated SVD|
|Gama et al. (2014) ACM Comput Surv 46(4):44|Concept drift, online learning|
|Fogel (2018) Contemp Clin Trials Commun 11:156–164|Clinical trial failure rates|
|Ghadessi et al. (2020) Orphanet J Rare Dis 15:184|Predictors of trial termination|
|Saito \& Rehmsmeier (2015) PLOS ONE 10(3):e0118432|PR curves for imbalanced data|
|Brier (1950) Mon Weather Rev 78:1–3|Proper scoring rules / calibration|

\---

## Author

**Sm Hasan ul Bari** — MBBS · MSc Advanced Biostatistics \& Epidemiology (Erasmus Mundus, Distinction) · MSc Health Economics \& Decision Modelling (Sheffield, Merit)  
Associate HTA Analyst, NICE Manchester · Technical Consultant, UNICEF NYC  
[github.com/sm-hasanulbari](https://github.com/sm-hasanulbari) · [ORCID: 0000-0002-5209-2029](https://orcid.org/0000-0002-5209-2029)

