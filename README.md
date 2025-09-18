# Surgical Skill Assessment Using Kinematic Features and Deep Learning on the JIGSAWS Dataset

This repository contains the implementation of our research project on **objective surgical skill assessment** using kinematic features and sequence-based deep learning models.  
We systematically compare interpretable handcrafted motion features with machine learning (ML) models and sequence-based deep learning (DL) approaches on the **JIGSAWS dataset**.

---

## 1. Project Overview

- **Goal**: To evaluate whether interpretable kinematic features can serve as reliable indicators of surgical expertise, and how they compare with black-box deep learning models.  
- **Dataset**: JIGSAWS (Knot-Tying, Needle-Passing, Suturing tasks).  
- **Features**:  
  - Basic: Path length, Straightness, Mean/Max velocity, Pause ratio, Frequency energy.  
  - Extended: Jerk mean/std, Bimanual coordination, Spectral bandpower (0.5–3 Hz, 3–6 Hz), Trajectory dispersion.  
- **Models**:  
  - Machine Learning: SVM (RBF kernel), XGBoost.  
  - Deep Learning: 2-layer LSTM.  
- **Evaluation**: Accuracy, Macro-F1, Ablation study, Feature importance analysis, Visualization of trajectories.

---

## 2. Dataset

The JIGSAWS dataset is not included in this repository due to size and license restrictions.  
You must download it manually from the official source:

➡ [JIGSAWS dataset download page](https://cirl.lcsr.jhu.edu/research/hmm/datasets/jigsaws_release/)

After downloading, place the dataset under the `data/` directory.

---

## 3. Repository Structure

```

Surgical-Skill-JIGSAWS
├── data/                     # Raw JIGSAWS data (to be downloaded separately)
├── out/                      # Output directory (figures, results, cache)
│   ├── ablation/             # Ablation study results
│   ├── featimp/              # Feature importance results
│   ├── featdist/             # Feature distributions (boxplots, violins)
│   ├── compare/              # Novice vs Expert trajectory comparisons
│   ├── dl/                   # Deep learning (LSTM) results
│   ├── ml\*/                  # Machine learning (SVM, XGB) results
│   ├── replay/               # 3D trajectory replay (gif/png)
│   └── figs/                 # Summary figures and tables
├── src/                      # Source code
│   ├── 00\_prepare.py         # Dataset preparation
│   ├── 01\_features.py        # Feature extraction
│   ├── 02\_ml\_baseline.py     # ML baseline models (SVM, XGB)
│   ├── 03\_dl\_models.py       # LSTM training and evaluation
│   ├── 04\_stats\_plots.py     # Statistical analysis and plots
│   ├── 05\_ablation.py        # Ablation experiments
│   ├── 06\_feat\_importance.py # Feature importance analysis
│   ├── 07\_summary\_figure.py  # Generate summary figure
│   ├── 08\_extra\_plots.py     # Additional visualization
│   ├── 09\_replay\_3d.py       # 3D trajectory replay
│   ├── 10\_compare\_trajectories.py # Novice vs Expert comparison
│   ├── 10\_pick\_trials.py     # Select representative trials
│   ├── 11\_feature\_distributions.py # Feature distribution plots
│   └── utils.py              # Utility functions
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
└── test.ipynb                # Example notebook

````

---

## 4. Installation

Create a new environment and install dependencies:

```bash
python3 -m venv cephalo-env
source cephalo-env/bin/activate
pip install -r requirements.txt
````

---

## 5. Usage

### Step 1. Prepare features

```bash
python src/01_features.py
```

### Step 2. Train ML baseline models (SVM, XGB)

```bash
python src/02_ml_baseline.py
```

### Step 3. Train DL model (LSTM)

```bash
python src/03_dl_models.py
```

### Step 4. Run ablation study

```bash
python src/05_ablation.py
```

### Step 5. Feature importance

```bash
python src/06_feat_importance.py
```

### Step 6. Visualizations

```bash
python src/11_feature_distributions.py
```

Outputs will be saved in the `out/` directory.

---

## 6. Results (Summary)

* **Macro-F1**:

  * XGBoost (basic features): 0.68
  * XGBoost (extended features): 0.83
  * LSTM: 0.89

* **Key Findings**:

  * Jerk (smoothness) and Bandpower 3–6 Hz (rhythmicity) are the most discriminative features.
  * Handcrafted interpretable features achieve performance close to deep learning models.
  * Provides explainability and educational value for surgical training and credentialing.

---

## 7. Citation

If you use this repository, please cite our work:

```
Oh S. Surgical Skill Assessment Using Kinematic Features and Deep Learning on the JIGSAWS Dataset.
Medical Metaverse Society Autumn Conference, 2025.
```

---

## 8. Contact

For questions or collaboration, please contact:

* **Author**: Seoyoung Oh
* **Affiliation**: Dept. of AI Big Data Convergence & Management, Kookmin University, Seoul, Republic of Korea
* **Email**: [5seo0\_oh@kookmin.ac.kr](mailto:5seo0_oh@kookmin.ac.kr)
* **GitHub**: [https://github.com/5seo0yong/Surgical-Skill-JIGSAWS](https://github.com/5seo0yong/Surgical-Skill-JIGSAWS)

---

