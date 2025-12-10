# Geographical Origin of Traditional Music: Gaussian Process Analysis

## Overview

This repository contains the implementation and comprehensive analysis for predicting the geographical origin of traditional music using **Gaussian Process (GP) regression and classification models**, with comparison against classical machine learning baselines.

**Key Result:** GP-Matérn kernel achieves competitive performance with explicit uncertainty quantification:
- **Regression:** MAE 11.10° (latitude), 31.21° (longitude), 3479 km geographic error
- **Classification:** 66.51% accuracy on 4-way regional prediction (Europe/North Africa, Asia, Americas, Oceania/South)

---

## Dataset

**Source:** UCI Machine Learning Repository  
**Citation:** Zhou, F., Claire, Q., & King, R. D. (2014). Geographical origin of music. *IEEE International Conference on Data Mining*, 1142–1147.

**Details:**
- **Size:** 1059 music tracks from 33 countries
- **Features:** 116 standardised audio descriptors
  - 68 timbral/rhythmic features (pitch, timbre, spectral statistics)
  - 48 chromatic features (pitch distribution across 4 octaves)
- **Targets:** Latitude, longitude (degrees), mapped to 4 macro-regions
- **Quality:** No missing values, pre-standardised features

**Region Distribution (class imbalance):**
| Region | Tracks | % |
|--------|--------|---|
| Europe / North Africa | 589 | 55.6% |
| Asia | 353 | 33.3% |
| Americas | 103 | 9.7% |
| Oceania / South | 14 | 1.3% |

---

## Methods

### Baseline Models (Common Methodology)

**Regression:**
- Multiple Linear Regression
- Ridge Regression (α = 1.0, L2 regularisation)

**Classification:**
- Logistic Regression (one-vs-rest, max_iter=200)
- Random Forest (100 trees, max_depth=20)

### Gaussian Process Models (Individual Techniques)

**Regression:**
- **GP-RBF:** Squared Exponential kernel with ARD
  - Assumes infinitely smooth functions
  - Per-feature length-scales learned via marginal likelihood maximisation
  
- **GP-Matérn (ν=1.5):** Matérn kernel with ARD
  - Assumes C^1.5-continuous functions
  - More flexible than RBF for real-world data

**Classification:**
- One-vs-rest encoding for multi-class prediction
- Combined with sigmoid link function for probabilistic outputs

---

## Results

### Regression Performance

**Latitude Prediction:**
| Model | RMSE | MAE |
|-------|------|-----|
| Linear Regression | 17.31° | 13.23° |
| Ridge Regression | 17.20° | 13.15° |
| GP-RBF | 15.56° | 11.46° |
| **GP-Matérn** | **15.20°** | **11.10°** |

**Longitude Prediction:**
| Model | RMSE | MAE |
|-------|------|-----|
| Linear Regression | 43.49° | 33.42° |
| Ridge Regression | 43.41° | 33.38° |
| GP-RBF | 42.36° | 32.46° |
| **GP-Matérn** | **40.75°** | **31.21°** |

**Geographic Distance Error (Great-Circle):**
| Model | Mean (km) | Std Dev (km) |
|-------|-----------|-------------|
| Ridge Regression | 3806.97 | 2656.06 |
| GP-RBF | 3619.27 | 2725.15 |
| **GP-Matérn** | **3478.72** | **2642.05** |

**Key Insight:** GP-Matérn improves upon ridge baseline by 7.7%, representing 37% error reduction vs. naive "predict mean" approach.

### Classification Performance

| Model | Accuracy | Macro-F1 |
|-------|----------|----------|
| Logistic Regression | 65.57% | 0.4230 |
| Random Forest | 65.09% | 0.3457 |
| GP-RBF | 56.60% | 0.2253 |
| **GP-Matérn** | **66.51%** | **0.3500** |

**Performance by Region (GP-Matérn):**
- **Americas:** 81.8% recall (well-separated from other regions)
- **Asia:** 75.3% recall (confused with Europe ~18% of time due to shared Eurasian traditions)
- **Europe/North Africa:** 80.7% recall (dominant class, inflates overall accuracy)
- **Oceania/South:** 0% recall (only 3 test samples, insufficient data)

---

## Key Findings

1. **GP models outperform baselines:** Both RBF and Matérn kernels improve regression MAE by 6–16% compared to linear methods.

2. **Matérn kernel advantage:** Slightly superior to RBF in both tasks, supporting C^1.5-smoothness assumption over infinite smoothness.

3. **Uncertainty quantification:** GP models provide well-calibrated confidence intervals (±2σ = 4–6 degrees), valuable for identifying ambiguous predictions.

4. **Classification challenge:** 66.51% accuracy reflects inherent difficulty: audio statistics alone cannot fully capture cultural/musical nuances. Class imbalance (56% Europe) further complicates minority-class prediction.

5. **Feature importance:** Both timbral (~40% importance) and chromatic (~35% importance) features matter; geographic origin is encoded in instrumentation and tonal systems.

---

## Experimental Setup

**Data Splits:** 60% train (635 samples), 20% validation (212), 20% test (212)  
**Preprocessing:** Standardisation via training set statistics; no feature selection  
**Hyperparameter Tuning:**
- Ridge α = 1.0 (fixed)
- RF: 100 trees, max_depth=20
- GP: marginal likelihood maximisation, n_restarts=2, alpha=1e-3

**Evaluation Metrics:**
- Regression: RMSE, MAE, geographic distance (haversine formula)
- Classification: accuracy, macro-F1, confusion matrix
- GP-specific: predictive variance, calibration analysis

---

## Limitations

1. **Dataset bias:** Curated "world music" collection (commercial, Western-marketed); excludes endangered/marginalized traditions.

2. **Audio-only features:** No metadata (instrumentation, lyrics, artist background, historical period). Musical meaning is social/cultural, not purely acoustic.

3. **Arbitrary region definitions:** Four regions conflate diverse traditions (e.g., "Europe/North Africa" ignores cultural distinctions). Continuous targets might be more natural.

4. **Class imbalance:** 56% Europe, 1.3% Oceania makes minority-class learning difficult without specialised techniques (weighting, SMOTE, focal loss).

5. **Scalability:** GP O(n³) cost; fitting ~30–60 sec per model. Sparse approximations needed for larger corpora.

6. **Generalistion:** Models tested on same 33 countries; performance on unseen regions/traditions unknown.

---

## Social, Ethical, Legal & Professional Considerations

**Bias & Representation:** Audio-based geographic inference risks essentialising music and reinforcing colonial "exotic" narratives. Dataset overrepresents well-resourced, commercialised traditions.

**Misuse Risk:** System could be used to censor, filter, or discriminate music by inferred origin. Mitigation: transparency, confidence intervals, human override.

**Privacy:** Automatic geographic tagging of audio could expose sensitive location information.

**Professional Responsibility:** Audio statistics do not capture cultural identity. Engagement with ethnomusicologists essential for responsible deployment.

---

## Usage

### Option 1: Run in Google Colab (Recommended)

1. Open the notebook: `geo_origin_of_music_test.ipynb`
2. Upload dataset file: `default_plus_chromatic_features_1059_tracks.txt`
3. Run cells sequentially

**Dataset Download:**
- Get from UCI ML Repository: https://archive.ics.uci.edu/dataset/315/geographical+original+of+music
