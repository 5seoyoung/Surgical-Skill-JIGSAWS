# Surgical-Skill-JIGSAWS

Surgical Skill Assessment Using Kinematic Features and Deep Learning on the JIGSAWS Dataset  
Oh Seoyoung (Kookmin University)

---

## 1. Description

This repository implements a systematic study on **objective surgical skill assessment** using the JIGSAWS dataset.  
We benchmarked **handcrafted kinematic features** against **sequence-based deep learning models**, highlighting both performance and interpretability.

- **Why important?**  
  Objective and automated evaluation of surgical skills is crucial for education, feedback, and credentialing [1].  
  Deep learning models have achieved strong performance, but their **black-box nature** limits interpretability, reproducibility, and trust in clinical adoption [2].  
  In contrast, **kinematic features** provide explainable evidence (smoothness, rhythmicity, efficiency) that can be directly connected to surgical pedagogy and training.

- **Our contribution:**  
  - Verified which motion descriptors (e.g., smoothness, rhythmicity) best separate skill levels.  
  - Showed that interpretable features can approximate deep learning accuracy while offering transparency.  
  - Proposed a pathway for integrating explainable metrics into **surgical training, feedback systems, and certification frameworks**.

---

## 2. Dataset

- **JIGSAWS** (Knot-Tying, Needle-Passing, Suturing)  
- Robotic kinematic data (76 dimensions)  
- Sliding window segmentation: 10s (stride 1s)  
- Labels: Novice (87), Intermediate (65), Expert (52) → total 204 samples  
- Download: [JIGSAWS official site](https://cirl.lcsr.jhu.edu/research/hmm/datasets/jigsaws_release/)  

Dataset is not included in this repository due to licensing and size restrictions.  
Users must **manually download** it from the official source above.

---

## 3. Methods

- **Preprocessing**  
  - Time alignment, outlier removal, resampling @ 50 Hz  
  - Tooltip coordinate extraction, z-score normalization per trial  

- **Feature Engineering**  
  - **Basic (6):** Path length, Straightness, Mean/Max velocity, Pause ratio, Frequency energy  
  - **Extended (6):**  
    - `jerk_mean`, `jerk_std`: smoothness (fine vibration)  
    - `bandpower (0.5–3 Hz, 3–6 Hz)`: rhythmicity of motion  
    - `disp_std`: spatial dispersion (stability)  
    - `bimanual_corr`: degree of coordination between both hands  

- **Modeling**  
  - Machine Learning: SVM (RBF), XGBoost  
  - Deep Learning: 2-layer LSTM (sequence input)  
  - Class imbalance handled by class weights & subsampling  

- **Evaluation**  
  - Metrics: Accuracy, Macro-F1  
  - Ablation study for feature contribution  
  - Permutation importance (macro-F1 drop)  
  - Group-level differences: Kruskal–Wallis test  
  - Visualization: XY density maps, 3D trajectories  

---

## 4. Results

- **Performance**  
  - XGBoost (basic features): Macro-F1 = 0.68  
  - XGBoost (extended features): Macro-F1 = **0.83**  
  - LSTM (raw sequence): Macro-F1 = **0.89**  

- **Ablation Study (XGB)**  
  - Removing jerk → ΔF1 −0.12 (largest drop)  
  - Removing bandpower → ΔF1 −0.06  
  - Removing bimanual_corr → negligible (ΔF1 −0.006)  

- **Feature Importance**  
  - Top: `jerk_mean (0.169)`, `bandpower 3–6Hz (0.095)`  
  - Secondary: `path_len (0.072)`, `max_v (0.022)`  
  - Minimal: `straight`, `mean_v` (≈ 0.000)  

- **Key Insight**  
  - **Smoothness (jerk) and Rhythmicity (bandpower)** are the most robust markers of surgical expertise.  
  - Interpretable features alone can nearly match deep learning, enabling **transparent, reproducible, and educationally relevant assessment**.

---

## 5. Discussion

- This study provides a **practical and explainable approach** for surgical skill evaluation.  
- Insights:  
  - Smoothness (jerk) → differentiates unstable novice motion vs. stable expert motion  
  - Rhythmicity (bandpower) → highlights periodic, efficient expert patterns  
- Implications:  
  - Can guide **curriculum design** for surgical education  
  - Useful for **feedback systems** to provide interpretable metrics  
  - Potential to support **objective credentialing frameworks** in clinical practice  

---

## 6. References

[1] Martin, J. A., Regehr, G., Reznick, R., MacRae, H., Murnaghan, J., Hutchison, C., & Brown, M. (1997). Objective structured assessment of technical skill (OSATS) for surgical residents. *The British Journal of Surgery*, 84(2), 273–278.  
[2] Hung, A. J., Chen, J., Gill, I. S., & 9 others. (2019). Deep learning using automated surgical performance analysis: Technical and methodological considerations. *European Urology Focus*, 5(5), 837–839.  

---

## 7. Repository Structure

```

Surgical-Skill-JIGSAWS
├── data/                 # dataset directory (empty, user must download JIGSAWS)
├── src/                  # source code
│   ├── 00\_prepare.py
│   ├── 01\_features.py
│   ├── 02\_ml\_baseline.py
│   ├── 03\_dl\_models.py
│   ├── ...
├── out/                  # experiment outputs
│   ├── ablation/
│   ├── featimp/
│   ├── featdist/
│   ├── dl/
│   ├── ml/
│   └── figs/
└── README.md

```

---

## 8. Citation

If you use this code, please cite:

```

@misc{oh2025surgicalskill,
author       = {Oh, Seoyoung},
title        = {Surgical Skill Assessment Using Kinematic Features and Deep Learning on the JIGSAWS Dataset},
year         = {2025},
publisher    = {GitHub},
howpublished = {\url{[https://github.com/5seoyoung/Surgical-Skill-JIGSAWS}}](https://github.com/5seoyoung/Surgical-Skill-JIGSAWS}})
}

```

---
