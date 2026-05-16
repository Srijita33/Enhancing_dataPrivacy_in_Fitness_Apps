# Enhancing_dataPrivacy_in_Fitness_Apps
# Privacy-Preserving Fitness Tracking Framework

Algorithmic framework for enhancing user data privacy in fitness tracking ecosystems using classical anonymization techniques and advanced hybrid anonymization models.

This project evaluates the effectiveness of:

* K-Anonymity
* L-Diversity
* T-Closeness
* Hybrid Model 1 (Adaptive Distribution Balancing)
* Hybrid Model 2 (Clustering-Aware Harmonization)
* Hybrid Model 3 (Class-Cap + Entropy + Stochastic Defense)

The proposed Hybrid Model 3 combines:

* Class-cap enforcement (≤55%)
* Entropy maximization
* Seeded stochastic selection
* Global distribution lock

and reduces inference attack accuracy from **67.93%** (T-Closeness baseline) to **54.41%**.

---

# Research Objective

Wearable fitness devices continuously generate physiological and behavioral data. Even after anonymization, adversaries can infer sensitive health status from quasi-identifiers such as age, region, and activity type.

This project develops an attack-aware anonymization framework for privacy-preserving data publishing in fitness tracking ecosystems.

The framework specifically targets:

* Derived attribute disclosure
* Majority-class inference attacks
* Distributional imbalance within equivalence classes

---

# Sensitive Attribute

The sensitive attribute used in this project is:

```text
fitness_level ∈ {fit, moderately_fit, unfit}
```

Derived deterministically using:

```text
IF steps ≥ 12000 AND calories ≥ 600 → fit
IF steps < 7000 AND calories < 350 → unfit
ELSE → moderately_fit
```

---

# Dataset Information

Synthetic wearable fitness dataset:

* Total Records: 7,000
* Zero Data Loss Across All Stages
* Utility Score: 1.000

## Quasi-Identifiers

* age
* region
* activity_type

## Numerical Attributes

* heart_rate
* steps
* calories
* sleep_hours
* weight_kg

## Sensitive Attribute

* fitness_level

---

# Privacy Pipeline

```text
Raw Data
   ↓
K-Anonymity
   ↓
L-Diversity
   ↓
T-Closeness
   ↓
Hybrid Model 1
   ↓
Hybrid Model 2
   ↓
Hybrid Model 3
```

Each stage receives the output of the previous stage, enabling cumulative privacy enhancement.

---

# Implemented Privacy Models

## 1. K-Anonymity

* k = 5
* Generalization-based anonymization
* Prevents direct identity linkage
* Does not protect against inference attacks

### Result

* Attack Accuracy: 68.97%

---

## 2. L-Diversity

* l = 3
* Ensures all sensitive values appear in equivalence classes
* Reduces homogeneity attacks
* Still vulnerable to skewness attacks

### Result

* Attack Accuracy: 68.61%

---

## 3. T-Closeness

* Uses Total Variation Distance (TVD)
* Distributional privacy preservation
* Threshold: t = 0.25

### Result

* Attack Accuracy: 67.93%

---

## 4. Hybrid Model 1 — Adaptive Distribution Balancing

Features:

* Violation-proportional correction
* Adaptive TVD balancing
* ξ-based correction strength

### Result

* Attack Accuracy: 65.63%

---

## 5. Hybrid Model 2 — Clustering-Aware Harmonization

Features:

* Age-cluster-aware target distributions
* Young / Middle / Senior partitioning
* Physiologically realistic balancing

### Result

* Attack Accuracy: 64.34%

---

## 6. Hybrid Model 3 — Proposed Final Framework

Core mechanisms:

### Pass 1 — Class-Cap Enforcement

* Maximum class proportion ≤ 55%
* Directly bounds majority-class adversary accuracy

### Pass 2 — Entropy-Guided Redistribution

* Shannon entropy maximization
* Stochastic boundary reassignment
* Seeded randomness for reproducibility

### Pass 3 — Global Distribution Lock

* Prevents excessive global distribution drift
* Maintains population-level statistical validity

### Final Result

* Attack Accuracy: 54.41%
* Reduction from T-Closeness: 13.52 percentage points
* Utility Score: 1.000
* Zero Data Loss

---

# Comparative Results

| Method         | Attack Accuracy | Mean TVD | Utility |
| -------------- | --------------- | -------- | ------- |
| K-Anonymity    | 68.97%          | 0.281    | 1.000   |
| L-Diversity    | 68.61%          | 0.283    | 1.000   |
| T-Closeness    | 67.93%          | 0.262    | 1.000   |
| Hybrid Model 1 | 65.63%          | 0.238    | 1.000   |
| Hybrid Model 2 | 64.34%          | 0.199    | 1.000   |
| Hybrid Model 3 | 54.41%          | 0.162    | 1.000   |

---

# Key Contributions

* Developed a privacy-preserving fitness data publishing framework
* Introduced attack-aware anonymization evaluation
* Implemented progressive 6-stage anonymization pipeline
* Designed Hybrid Model 3 for majority-class inference suppression
* Achieved strong privacy improvement with zero numeric utility loss

---

# Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Jupyter Notebook

---

# Project Structure

```text
notebooks/
├── 01_data_generation.ipynb
├── 02_k_anonymity.ipynb
├── 03_l_diversity.ipynb
├── 04_t_closeness.ipynb
├── 07_hybrid1_adaptive_microagg.ipynb
├── 08_hybrid2_clustering_aware.ipynb
├── 09_hybrid3_enhanced.ipynb
├── 10_final_comparison.ipynb

results/

src/

data/
```

---

# Research Outcome

The study demonstrates that:

* Structural privacy guarantees alone are insufficient
* Inference attack resistance must be explicitly modeled
* Hard class-cap constraints outperform soft distributional constraints
* Strong privacy can be achieved without sacrificing numeric utility

---

# Author

Srijita Ghosh

School of Computer Science and Engineering
VIT Vellore

---

# Reference

Research Presentation:

Privacy-Preserving Data Publishing for Wearable Health Data fileciteturn0file0
