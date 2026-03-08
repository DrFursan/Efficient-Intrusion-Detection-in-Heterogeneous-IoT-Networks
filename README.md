# AEGIS-IoT: Adaptive Ensemble Framework for Intrusion Detection in Heterogeneous IoT Networks

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange)](https://scikit-learn.org)

> Official implementation for the paper:  
> **"AEGIS-IoT: A High-Stability Adaptive Ensemble Framework for Efficient Intrusion Detection in Heterogeneous IoT Networks"**

---

## Overview

AEGIS-IoT is a hybrid ensemble machine learning framework for network intrusion detection in IoT environments. The framework evaluates six individual classifiers and three ensemble strategies across two feature-selection scenarios on the **UNSW-NB15** and **BoT-IoT** datasets.

### Key Results

| Model | Accuracy | F1-Score | Time (s) |
|-------|----------|----------|----------|
| DT+LR Stacking (Scenario 2) | **1.000** | **1.000** | **29.91** |
| RF+SVM Voting (Scenario 2) | 1.000 | 1.000 | 409.43 |
| SVM+GB Voting (Scenario 2) | 1.000 | 1.000 | 2,263.97 |
| Random Forest (Scenario 1) | 1.000 | 0.995 | 818.76 |
| Decision Tree (Scenario 1) | 1.000 | 0.995 | 34.62 |

The **DT+LR Stacking** ensemble with RF-selected features achieves perfect classification while being **76× faster** than the SVM+GB voting ensemble.

---

## Repository Structure

```
AEGIS-IoT/
├── README.md
├── requirements.txt
├── .gitignore
├── run_all.py                  # Master pipeline — runs everything end to end
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py        # Data loading, cleaning, encoding, scaling
│   ├── feature_selection.py    # Scenario 1 (IG/MI) and Scenario 2 (RF importance)
│   ├── evaluate.py             # Shared metrics, confusion matrix, CSV export
│   ├── visualize.py            # All publication figures (10 charts)
│   │
│   └── models/
│       ├── __init__.py
│       ├── train_individual.py # RF, DT, SVM, NB, LR, GB, XGBoost
│       ├── train_stacking.py   # DT + LR stacking ensemble
│       └── train_voting.py     # RF+SVM and SVM+GB hard voting ensembles
│
├── data/
│   └── README.md               # Dataset download instructions
│
├── results/                    # Auto-generated CSVs and .txt reports
└── figures/                    # Auto-generated PNG figures
```

---

## Datasets

This project uses two publicly available IoT network traffic datasets.

### UNSW-NB15
- **Source:** <https://research.unsw.edu.au/projects/unsw-nb15-dataset>
- Download the training and testing CSV files.
- Place them in `data/` as `UNSW_NB15_training-set.csv` and `UNSW_NB15_testing-set.csv`.

### BoT-IoT
- **Source:** <https://research.unsw.edu.au/projects/bot-iot-dataset>
- Download the full feature CSV files.
- Place the combined file in `data/` as `BoT_IoT_Dataset.csv`.

See `data/README.md` for detailed instructions.

---

## Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/AEGIS-IoT.git
cd AEGIS-IoT

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

### Run the Full Pipeline

```bash
python run_all.py
```

This executes all steps in order:
1. Preprocessing and cleaning
2. Feature selection (both scenarios)
3. Training all individual classifiers
4. Training all ensemble models
5. Evaluating and saving results to `results/`
6. Generating all figures to `figures/`

### Run Individual Steps

```bash
# Preprocessing only
python src/preprocessing.py --input data/UNSW_NB15_testing-set.csv

# Feature selection
python src/feature_selection.py --scenario 1   # IG + Mutual Information
python src/feature_selection.py --scenario 2   # RF importance

# Train a specific model
python src/models/train_individual.py --model rf       # Random Forest
python src/models/train_individual.py --model dt       # Decision Tree
python src/models/train_individual.py --model svm      # SVM
python src/models/train_individual.py --model nb       # Naive Bayes
python src/models/train_individual.py --model lr       # Logistic Regression
python src/models/train_individual.py --model gb       # Gradient Boosting
python src/models/train_individual.py --model xgb      # XGBoost

# Train ensemble models
python src/models/train_stacking.py            # DT + LR stacking
python src/models/train_voting.py --combo rf_svm   # RF + SVM voting
python src/models/train_voting.py --combo svm_gb   # SVM + GB voting

# Generate all figures
python src/visualize.py
```

---

## Experimental Scenarios

| | Scenario 1 | Scenario 2 |
|--|--|--|
| **Feature Selection** | Information Gain + Mutual Information | Random Forest Importance |
| **No. of Features** | Top 10 (combined IG + MI) | Top 9 RF-ranked features |
| **Dataset** | UNSW-NB15 (stratified 5% subset) | UNSW-NB15 (combined resampled) |
| **Purpose** | Baseline with domain-guided selection | Automated data-driven selection |

---

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{thabit2025aegisiot,
  author  = {Thabit, Fursan and Can, Ozgu and Abd-Rabbou, M. Y. and
             Alkhater, Wadah and Zhang, Yuqing},
  title   = {{AEGIS-IoT}: A High-Stability Adaptive Ensemble Framework
             for Efficient Intrusion Detection in Heterogeneous {IoT}
             Networks},
  journal = {<journal name>},
  year    = {2025}
}
```

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## Acknowledgements

- UNSW-NB15 and BoT-IoT datasets courtesy of the Cyber Range Lab, UNSW Canberra.
- Built with [scikit-learn](https://scikit-learn.org), [imbalanced-learn](https://imbalanced-learn.org), [XGBoost](https://xgboost.readthedocs.io), and [matplotlib](https://matplotlib.org).
