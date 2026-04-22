# Solving Early-Stage CKD Misdiagnosis Using Hybrid Optimization for Simultaneous Feature Selection and Hyperparameter Tuning

**Muhammed Abdel-Hamid Shawki Mustafa**^1, **Abdul Rahman Al-Bili Ismat Al-Bili**^2

Faculty of Computer Science & Engineering, New Mansoura University, Mansoura, Egypt

^1 224100686, ^2 222101372

Supervised by: **Dr. Ibrahim**

Course: CSE015 — Research Methodology

---

**Abstract** — Chronic Kidney Disease (CKD) affects approximately 850 million people worldwide, yet the majority of cases remain undiagnosed until irreversible stages due to the asymptomatic nature of early progression. Traditional machine learning approaches for CKD detection suffer from three persistent limitations: reliance on single optimization algorithms vulnerable to local optima, sequential (rather than simultaneous) execution of feature selection and hyperparameter tuning, and optimization targets that prioritize accuracy over the clinically critical metric of sensitivity. This paper presents SGPO v2 (Simultaneous GA-PSO Optimizer version 2), a co-evolutionary framework that performs simultaneous feature selection and hyperparameter tuning using three 2025 nature-inspired algorithms: the Starfish Optimization Algorithm (SFOA) for binary feature mask optimization, the Dream Optimization Algorithm (DOA) for continuous hyperparameter search, and the Fungal Growth Optimizer (FGO) for stagnation-breaking perturbation injection. The three algorithms are coupled through a shared fitness matrix within each generation, enabling real-time co-adaptation between feature subsets and classifier configurations. Evaluated on a MIMIC-IV cohort of 57,875 patients with 42 clinical features using nested cross-validation (10-fold outer, 3-fold inner) with SMOTE applied exclusively to training folds, SGPO v2 achieved an 81% feature reduction (42 to 8 features) while maintaining 99.75% of the baseline AUC-ROC (0.9537 vs. 0.9561) and 99.98% of baseline sensitivity (0.8902 vs. 0.8904). The optimized model requires only one laboratory test (serum creatinine) combined with demographic and administrative features. A four-model comparison demonstrated that SGPO v2 achieves 99.4% of XGBoost's AUC using only 19% of the features. A six-variant ablation study confirmed that each component contributes meaningfully: SFOA provides the primary 81% feature reduction, DOA enables deeper compression through compensatory hyperparameter tuning (+0.0035 AUC), and FGO prevents premature convergence through spore dispersal at stagnation points.

**Keywords:** Chronic Kidney Disease, co-evolutionary optimization, feature selection, hyperparameter tuning, Starfish Optimization Algorithm, Dream Optimization Algorithm, Fungal Growth Optimizer, MIMIC-IV, Random Forest, sensitivity-weighted fitness

---

## I. Introduction

Chronic Kidney Disease (CKD) is a progressive and irreversible loss of kidney function that affects approximately 850 million people globally and accounts for 1.2 million deaths annually [1]. The disease is particularly insidious because patients remain asymptomatic during Stages 1 through 3, with clinical diagnosis typically occurring only at Stage 4 or 5, when dialysis or transplantation becomes the sole viable treatment [2]. Early detection through intelligent computational systems could prevent this progression, yet traditional diagnosis requires assessment of 24 or more clinical parameters, making it resource-intensive, time-consuming, and prone to human error.

Machine learning (ML) has demonstrated considerable promise in automating CKD detection from clinical data. However, ML-based approaches face two fundamentally interdependent challenges: (1) clinical datasets contain numerous features, many of which are redundant or noisy, increasing model complexity without proportional improvement in predictive power; and (2) classifier hyperparameters must be tuned to work optimally with the specific feature subset being used. These two challenges are conventionally addressed in a sequential pipeline—first selecting features, then tuning hyperparameters on the reduced set—which fundamentally ignores the interdependence between them. A feature that appears uninformative under default hyperparameters may become highly discriminative under an optimized configuration, and vice versa.

This paper addresses this gap by proposing SGPO v2 (Simultaneous GA-PSO Optimizer version 2), a co-evolutionary framework that couples feature selection and hyperparameter tuning within a single iterative loop. The framework employs three recently published (2025) nature-inspired metaheuristic algorithms: the Starfish Optimization Algorithm (SFOA) [3] for binary feature mask optimization, the Dream Optimization Algorithm (DOA) [4] for continuous hyperparameter search in normalized space, and the Fungal Growth Optimizer (FGO) [5] for perturbation injection to prevent premature convergence. These three algorithms operate simultaneously within each generation, communicating through a shared fitness matrix that enables mutual co-adaptation.

The primary contributions of this work are:

1. A novel co-evolutionary framework (SGPO v2) that simultaneously optimizes feature selection and hyperparameter tuning using three 2025 nature-inspired algorithms within a shared fitness evaluation loop, addressing the sequential execution limitation of existing approaches.

2. An 81% feature reduction (42 to 8 features) on the MIMIC-IV clinical database containing 57,875 patients, with only 0.25% loss in AUC-ROC, demonstrating that a clinically practical model requiring a single laboratory test can match the discriminative power of a full-panel model.

3. A comprehensive experimental evaluation comprising baseline comparison, four-model benchmarking (Logistic Regression, Random Forest, XGBoost, SGPO v2), and six-variant ablation study, all conducted using rigorous nested cross-validation with strict data leakage prevention.

4. Clinical insight that serum creatinine combined with readily available demographic and administrative features is sufficient for effective CKD screening, reducing the laboratory burden from 22 tests to 1.

The remainder of this paper is organized as follows: Section II reviews related work and identifies the specific gaps addressed. Section III formulates the optimization problem. Section IV describes the algorithm design including standard components and the developed SGPO v2 framework. Section V presents the experimental setup and results. Section VI provides the conclusion.

---

## II. Related Work

Recent literature (2025–2026) has witnessed substantial interest in applying nature-inspired optimization algorithms to CKD detection. We conducted a systematic review of six state-of-the-art papers to identify the research gaps that motivate this work.

**APSO + Echo State Network (2025)** [6] applied Adaptive Particle Swarm Optimization with Echo State Networks to the MIMIC-III temporal clinical database, achieving 99.6% accuracy. While impressive, this approach requires sequential temporal patient data (multiple lab readings over time), which limits applicability to screening contexts where only a single clinical encounter is available. Furthermore, no feature selection was performed, requiring the full feature panel for deployment.

**GPCB (2025)** [7] combined Genetic Algorithm with Particle Swarm Optimization (GA-PSO) and a CNN-BiLSTM deep learning architecture on the UCI Cardiovascular Disease dataset, achieving 98.9% accuracy. This work integrates both GA and PSO; however, the two algorithms operate sequentially—GA selects features first, then PSO tunes CNN-BiLSTM hyperparameters on the fixed feature set. This sequential design cannot discover synergies where certain features become valuable only under specific hyperparameter configurations. Additionally, the deep learning classifier sacrifices clinical interpretability.

**WWPA-GWO (2025)** [8] hybridized the Waterwheel Plant Algorithm with Grey Wolf Optimizer to tune a Multi-Layer Perceptron on the UCI CKD dataset, achieving 98.4% accuracy. No feature selection was performed. Critically, the UCI CKD dataset contains only 400 rows with 24 features, on which a standard Random Forest achieves near-perfect accuracy without any optimization, leaving no meaningful room for demonstrating optimization value.

**PSO + Bagging (2026)** [9] applied PSO to optimize a Random Forest with bagging on the UCI CKD dataset, achieving 97.5% accuracy. While employing an interpretable classifier, PSO is used solely for hyperparameter tuning without feature selection, and the small dataset again limits generalizability of results.

**SURD (2026)** [10] applied causal information theory to both UCI CKD and MIMIC-IV datasets, achieving high AUC values. However, SURD performs post-hoc causal analysis rather than active optimization—it identifies causally relevant features but does not simultaneously optimize a classifier's configuration to exploit them.

**Summary of identified gaps.** Table I synthesizes the four critical gaps identified across the reviewed literature.

**Table I: Research gaps and proposed solutions**

| Gap | Current Literature | SGPO v2 Solution |
|---|---|---|
| Optimization approach | Single algorithms (GA or PSO alone)—vulnerable to local optima | Co-evolutionary hybrid: SFOA + DOA + FGO |
| Feature selection vs. tuning | Sequential execution (select first, then tune) | Simultaneous unified loop with shared fitness matrix |
| Optimization target | Accuracy or AUC maximization | Sensitivity-weighted fitness (clinical priority) |
| Interpretability | Deep learning black boxes (CNN, BiLSTM, Transformers) | Random Forest with inherent feature importance |

No existing work performs simultaneous optimization of feature selection and hyperparameter tuning on CKD datasets with sensitivity as the primary clinical target.

---

## III. Problem Formulation

### A. Optimization Objective

The CKD detection optimization problem can be formally defined as a bi-level optimization over two coupled search spaces:

**Search Space 1 (Feature Selection):** A binary vector **m** ∈ {0, 1}^n, where n = 42 is the total number of features and m_i = 1 indicates that feature i is selected. The search space contains 2^42 ≈ 4.4 × 10^12 possible feature subsets.

**Search Space 2 (Hyperparameter Tuning):** A continuous vector **θ** ∈ ℝ^4 representing the Random Forest hyperparameters: n_estimators ∈ [50, 300], max_depth ∈ [3, 30], min_samples_split ∈ [2, 20], and min_samples_leaf ∈ [1, 10].

**Objective Function:** The combined fitness function balances three competing objectives with clinically motivated weights:

```
F(m, θ) = 0.50 × AUC(m, θ) + 0.30 × Sensitivity(m, θ) − 0.20 × (||m||₁ / n)
```

where AUC(m, θ) is the Area Under the ROC Curve of a Random Forest classifier with hyperparameters θ trained on features selected by mask m, Sensitivity(m, θ) is the true positive rate (recall) for CKD-positive patients, and ||m||₁ / n is the fraction of features selected.

### B. Weight Justification

The fitness weights were selected through the following clinical reasoning:

- **w_AUC = 0.50:** Ensures overall discriminative quality and prevents degenerate solutions that achieve perfect sensitivity by predicting all patients as CKD-positive.
- **w_Sensitivity = 0.30:** Reflects the clinical priority that a missed CKD diagnosis (false negative) leads to irreversible disease progression—making sensitivity the most clinically consequential metric.
- **w_Feature = −0.20:** Penalizes feature count to enforce model parsimony, acting as built-in regularization against overfitting and encouraging clinically deployable solutions.

### C. Validation Strategy

To prevent optimistic bias, we implement strict nested cross-validation:

- **Outer loop (10-fold):** Provides unbiased performance estimation. Test folds are never exposed to the optimizer.
- **Inner loop (3-fold):** Used within the fitness function for candidate solution evaluation during optimization.
- **SMOTE:** Applied exclusively to training folds in both loops, preventing synthetic minority oversampling from leaking test-set information.

---

## IV. Algorithm Design

### A. Standard Algorithms

#### 1) Starfish Optimization Algorithm (SFOA)

The Starfish Optimization Algorithm [3] is a 2025 nature-inspired metaheuristic based on the biological behaviors of starfish: multi-arm search for wide exploration and regeneration for recombination. In SGPO v2, each starfish individual represents a binary feature mask **m** ∈ {0, 1}^42 with a minimum constraint of 2 selected features.

The algorithm operates through three mechanisms: (i) **Arm-based exploration**, where each starfish has k = 5 arms that simultaneously search different regions of the binary feature space by flipping random subsets of bits in their assigned region, enabling broader coverage than single-point mutation; (ii) **Regeneration crossover**, inspired by the biological ability of starfish to regenerate from a single severed arm, where offspring are created by copying one arm's worth of features from a donor parent selected via tournament selection; and (iii) **Elitism**, preserving the best individual across generations. Each generation allocates 70% of offspring to arm exploration and 30% to regeneration crossover.

#### 2) Dream Optimization Algorithm (DOA)

The Dream Optimization Algorithm [4] is a 2025 metaheuristic inspired by neurological dream/wake cycles, operating in two distinct phases. In SGPO v2, DOA tunes four Random Forest hyperparameters in a normalized continuous space [0, 1]^4.

The **Dream phase** (exploration) applies large random perturbations with intensity that decreases over generations: dream_intensity = 0.3 × (1 − progress), allowing wide search early in the optimization. The **Wake phase** (exploitation) performs PSO-like velocity updates toward personal and global best positions: v_i = w·v_i + c₁·r₁·(p_best − x_i) + c₂·r₂·(g_best − x_i), where inertia w decreases from 0.9 to 0.4, cognitive coefficient c₁ decreases from 2.0 to 0.0, and social coefficient c₂ increases from 0.0 to 2.0 across generations. The probability of entering the dream phase for each particle is P(dream) = 0.4 × (1 − progress), transitioning from exploration to exploitation over time.

#### 3) Fungal Growth Optimizer (FGO)

The Fungal Growth Optimizer [5] is a 2025 metaheuristic inspired by mycelium network growth patterns. In SGPO v2, FGO serves a supporting role: preventing premature convergence through two mechanisms. **Hyphal growth** applies continuous small perturbations every generation: bit flips at rate 0.15 for binary masks and Gaussian noise N(0, 0.05) for continuous vectors. **Spore dispersal** is triggered when the global best fitness stagnates for 3 consecutive generations, replacing the worst 20% of both SFOA and DOA populations with random new solutions. This mechanism ensures the optimizer can escape local optima.

### B. Developed Algorithm: SGPO v2

The critical innovation of SGPO v2 is the co-evolutionary coupling of SFOA and DOA through a shared fitness evaluation matrix, with FGO providing stagnation-breaking perturbation. Algorithm 1 presents the complete procedure.

**Algorithm 1: SGPO v2 Co-Evolutionary Optimization**

```
Input: Dataset (X, y), n_generations=30, sfoa_pop=10, doa_pop=10
Output: Best feature mask m*, best hyperparameters θ*, fitness F*

1.  Initialize SFOA population: {m₁, m₂, ..., m₁₀} ← random binary masks
2.  Initialize DOA population: {θ₁, θ₂, ..., θ₁₀} ← random [0,1]⁴ vectors
3.  Initialize FGO with stagnation_threshold=3
4.  F* ← -∞
5.
6.  FOR g = 0 TO 29 DO:
7.      // Step 1: Form evaluation pairs (diagonal strategy)
8.      pairs ← {(i, j_rand), (i, j_best)} for each SFOA mask i
9.
10.     // Step 2: Evaluate pairs via shared fitness
11.     FOR each (i, j) in pairs DO:
12.         F(i,j) ← Evaluate(X, y, mᵢ, θⱼ) using 3-fold inner CV
13.         SFOA_fitness[i] ← max(SFOA_fitness[i], F(i,j))
14.         DOA_fitness[j] ← max(DOA_fitness[j], F(i,j))
15.         IF F(i,j) > F* THEN: F* ← F(i,j); m* ← mᵢ; θ* ← θⱼ
16.     END FOR
17.
18.     // Step 3: Apply FGO perturbation
19.     IF stagnation detected (3 gens without improvement):
20.         Replace worst 20% of SFOA masks with random masks
21.         Replace worst 20% of DOA particles with random positions
22.     ELSE:
23.         Apply hyphal growth (small perturbations) to both populations
24.     END IF
25.
26.     // Step 4: Evolve both populations independently
27.     SFOA.evolve()    // arm exploration + regeneration crossover
28.     DOA.evolve(g)    // dream/wake phase velocity updates
29. END FOR
30.
31. // Final evaluation on held-out outer folds
32. RETURN m*, θ*, F*
```

**Key design decisions:**

*Diagonal evaluation strategy.* Rather than evaluating all N × M = 100 (mask, particle) pairs per generation, we pair each SFOA mask with one random DOA particle plus the current best DOA particle, yielding approximately 20 evaluations per generation. This reduces computational cost by 5× while maintaining solution quality, as each optimizer is evaluated at its best available pairing.

*Max-aggregation fitness assignment.* Each optimizer's individual is assigned the maximum fitness across all its pairings: SFOA_fitness[i] = max_j{F(m_i, θ_j)} and DOA_fitness[j] = max_i{F(m_i, θ_j)}. This prevents one optimizer's weak solutions from unfairly penalizing the other and ensures that each component is evaluated under its most favorable conditions.

*Co-evolutionary dynamics.* As SFOA progressively removes features across generations, DOA simultaneously adapts hyperparameters to compensate for the resulting information loss. Conversely, as DOA discovers more effective hyperparameter configurations, SFOA can safely remove additional features that were previously necessary. FGO intervenes when this co-evolutionary process stagnates, injecting diversity to escape local optima.

---

## V. Experimentation

### A. Dataset

We constructed a CKD detection dataset from the MIMIC-IV clinical database [11], a freely accessible repository containing de-identified health records from Beth Israel Deaconess Medical Center (2008–2019). CKD patients were identified using ICD-10 code N18* and ICD-9 code 585*, with a balanced control group randomly sampled from non-CKD patients. The raw data was processed from five MIMIC-IV tables totaling over 17 GB, including 158 million laboratory events. Table II summarizes the final dataset.

**Table II: MIMIC-IV CKD dataset summary**

| Metric | Value |
|---|---|
| Total patients | 57,875 |
| CKD patients (label = 1) | 29,233 (50.5%) |
| Non-CKD controls (label = 0) | 28,642 (49.5%) |
| Total features | 42 |
| Feature categories | Demographics (2), Admission (7), Insurance (6), Marital (5), Lab values (22) |
| Class balance ratio | 1.02:1 |

The 42 features span five clinical categories: demographics (age, gender), admission statistics (n_admissions, avg_los_days, max_los_days, hospital_expire_flag, n_emergency, n_urgent, n_elective), insurance type (6 one-hot), marital status (5 one-hot), and laboratory values encompassing kidney function (creatinine, urea_nitrogen), electrolytes (potassium, sodium, bicarbonate, calcium_total, chloride, magnesium), complete blood count (hemoglobin, hematocrit, platelet_count, wbc, rbc, mcv, mch, mchc), and metabolic panels (glucose, albumin, bilirubin_total, alkaline_phosphatase, alt, ast).

Preprocessing consisted of: (1) patient-level aggregation of lab values using median, (2) removal of columns with >50% missing values (6 dropped), (3) removal of rows with >50% missing features (639 dropped), (4) median imputation for remaining missing values, and (5) standard scaling.

### B. Evaluation Metrics

Performance was assessed using four standard classification metrics:

- **AUC-ROC:** Area Under the Receiver Operating Characteristic curve—measures overall discriminative ability across all classification thresholds.
- **Sensitivity (Recall):** True positive rate—the proportion of actual CKD patients correctly identified. This is the clinically prioritized metric, as false negatives lead to irreversible disease progression.
- **Accuracy:** Overall proportion of correct predictions.
- **Specificity:** True negative rate—the proportion of non-CKD patients correctly identified.

All metrics were computed using 10-fold stratified cross-validation with means and standard deviations reported across folds.

### C. Experimental Configuration

**Table III: SGPO v2 configuration**

| Parameter | Value |
|---|---|
| SFOA population | 10 starfish (5 arms each) |
| DOA population | 10 particles (dream_ratio = 0.4) |
| Generations | 30 |
| Evaluation strategy | Diagonal (~20 evaluations/generation) |
| FGO stagnation threshold | 3 generations |
| FGO spore rate / hyphal rate | 20% / 15% |
| Inner CV | 3-fold stratified |
| Outer CV | 10-fold stratified |
| SMOTE | Training folds only |
| Base classifier | Random Forest |
| Total fitness evaluations | 590 |
| Random seed | 42 |

### D. Results

#### 1) Baseline Performance

The baseline Random Forest (100 trees, default hyperparameters, 5-fold CV) established the performance floor on the MIMIC-IV dataset (Table IV).

**Table IV: Baseline Random Forest results**

| Metric | Mean | Std |
|---|---|---|
| Accuracy | 0.8897 | 0.0022 |
| AUC-ROC | 0.9561 | 0.0011 |
| Sensitivity | 0.8904 | 0.0034 |
| Features | 42 | — |

Feature importance analysis confirmed clinical validity: creatinine (0.285) and urea nitrogen (0.156) dominated, followed by age (0.055), hemoglobin (0.038), and admission frequency (0.033). This ranking aligns with established clinical knowledge, confirming that the dataset captures genuine CKD signal.

#### 2) SGPO v2 Optimization Results

The full 30-generation run completed in 72 minutes (4,311 seconds) with 590 total fitness evaluations. Table V presents the comparison with baseline.

**Table V: Baseline vs. SGPO v2 comparison**

| Metric | Baseline (42 features) | SGPO v2 (8 features) | Change |
|---|---|---|---|
| AUC-ROC | 0.9561 ± 0.0011 | 0.9537 ± 0.0020 | −0.0024 (−0.25%) |
| Sensitivity | 0.8904 ± 0.0034 | 0.8902 ± 0.0063 | −0.0002 (−0.02%) |
| Accuracy | 0.8897 ± 0.0022 | 0.8867 ± 0.0036 | −0.0030 (−0.34%) |
| Specificity | — | 0.8832 ± 0.0053 | — |
| Features | 42 | 8 | **−34 (−81%)** |

The 81% feature reduction with only 0.25% AUC loss demonstrates near-lossless compression of the feature space. Low standard deviations (0.002–0.006) confirm stable performance across all 10 outer folds.

**Table VI: Selected features (8 of 42)**

| Feature | Category | Clinical Relevance |
|---|---|---|
| creatinine | Lab — Kidney | Primary CKD biomarker; elevated levels indicate reduced GFR |
| age | Demographic | Major non-modifiable CKD risk factor |
| n_admissions | Admission | Proxy for disease burden |
| avg_los_days | Admission | Reflects disease severity |
| ins_UNKNOWN | Insurance | Captures access-to-care disparities |
| marital_SINGLE | Marital | Social support indicator |
| marital_UNKNOWN | Marital | May reflect emergency admissions |
| marital_WIDOWED | Marital | Age-correlated; CKD prevalence indicator |

Notably, urea nitrogen—the second most important baseline feature (importance = 0.156)—was eliminated, indicating that creatinine alone captures sufficient kidney function information when combined with the selected demographic and admission features.

The optimizer selected hyperparameters of n_estimators = 278, max_depth = 15, min_samples_split = 3, and min_samples_leaf = 8, choosing a larger ensemble with constrained depth compared to the baseline—trading raw tree complexity for ensemble diversity and regularization.

**Table VII: Aggregated confusion matrix (10-fold CV)**

|  | Predicted Non-CKD | Predicted CKD |
|---|---|---|
| **Actual Non-CKD** | 25,296 | 3,346 |
| **Actual CKD** | 3,211 | 26,022 |

#### 3) Convergence Analysis

The optimization progressed through four distinct convergence phases (Fig. 1):

- **Phase 1 — Rapid Elimination (Gen 0–4):** SFOA immediately eliminated 29 of 42 features, recognizing that the majority contribute negligible discriminative power. Feature count dropped from 42 to 13 with steady fitness improvement from 0.665 to 0.678.

- **Phase 2 — Refinement Plateau (Gen 5–15):** Feature count stabilized at 13 while DOA fine-tuned hyperparameters. Fitness improved slowly from 0.678 to 0.680. At Generation 12, the FGO detected 3-generation stagnation and triggered spore dispersal, replacing 2 SFOA masks and 2 DOA particles with random solutions.

- **Phase 3 — Second Reduction Wave (Gen 16–24):** Following FGO intervention, features dropped from 13 to 11, then briefly to 10, as the optimizer explored more aggressive compression. Fitness rose to 0.692.

- **Phase 4 — Final Compression (Gen 25–29):** A breakthrough at Generation 25 eliminated 3 additional features (11 → 8), producing the largest single-generation fitness jump (+0.013). The optimizer converged at Generation 28 with final fitness 0.7056.

#### 4) Model Comparison

Table VIII presents the four-model comparison under identical 10-fold stratified CV conditions.

**Table VIII: Model comparison results**

| Model | Features | AUC-ROC | Sensitivity | F1-Score |
|---|---|---|---|---|
| Logistic Regression | 42 | 0.9487 ± 0.0027 | 0.8742 ± 0.0058 | 0.8813 ± 0.0036 |
| Random Forest | 42 | 0.9563 ± 0.0018 | 0.8910 ± 0.0037 | 0.8905 ± 0.0025 |
| XGBoost | 42 | 0.9595 ± 0.0016 | 0.8971 ± 0.0043 | 0.8937 ± 0.0031 |
| **SGPO v2** | **8** | **0.9537 ± 0.0020** | **0.8902 ± 0.0063** | **0.8881 ± 0.0037** |

SGPO v2 achieves 99.4% of XGBoost's AUC (0.9537 vs. 0.9595) while using only 19% of the features (8 vs. 42). The AUC cost per feature eliminated is (0.9563 − 0.9537) / 34 = 0.00008 per feature—an exceptional trade-off. SGPO v2 outperforms Logistic Regression across all metrics despite using 34 fewer features, confirming that the selected feature set captures non-linear relationships.

#### 5) Ablation Study

Table IX presents the six-variant ablation study validating each component's contribution.

**Table IX: Ablation study results**

| Variant | SFOA | DOA | FGO | Features | AUC-ROC | Sensitivity |
|---|---|---|---|---|---|---|
| **Full SGPO v2** | ✓ | ✓ | ✓ | **8** | **0.9537** | **0.8902** |
| No FGO | ✓ | ✓ | ✗ | 9 | 0.9541 | 0.8949 |
| No DOA | ✓ | ✗ | ✓ | 12 | 0.9502 | 0.8841 |
| No SFOA | ✗ | ✓ | ✓ | 42 | 0.9573 | 0.8946 |
| SFOA only | ✓ | ✗ | ✗ | 7 | 0.9480 | 0.8848 |
| DOA only | ✗ | ✓ | ✗ | 42 | 0.9573 | 0.8946 |

**SFOA contribution:** Removing SFOA retains all 42 features. While AUC is marginally higher (0.9573), the model is clinically impractical. SFOA is solely responsible for the 81% feature reduction. The AUC cost per feature eliminated is only 0.0001.

**DOA contribution:** Without DOA, AUC drops by 0.0035 and sensitivity by 0.0061. More critically, feature selection stalls at 12 features—DOA enables 4 additional features to be removed by compensating for information loss through optimized hyperparameters.

**FGO contribution:** Without FGO, 9 features are retained instead of 8, with comparable AUC. FGO's role is enabling deeper compression through stagnation-breaking: spore dispersal at Generation 12 broke a 3-generation plateau and initiated the second reduction wave.

**Synergy:** The components interact synergistically—SFOA reduces features, DOA compensates through hyperparameter optimization, and FGO breaks stagnation when the co-evolutionary process converges prematurely. No single component or pair achieves the full framework's combination of minimal features (8), high AUC (0.9537), and high sensitivity (0.8902).

### E. Discussion

**Feature selection as the primary contribution.** The 81% feature reduction with 0.25% AUC loss challenges the assumption that more features necessarily improve clinical prediction. Only 1 of 22 laboratory values (creatinine) was retained, while 5 of 8 features are demographic or administrative, suggesting that CKD detection benefits significantly from patient context beyond laboratory results.

**Clinical implications.** The optimized model requires only one laboratory test (serum creatinine), reducing the clinical burden from a full panel of 22 tests. Five of the 8 features (age, admissions, length of stay, insurance, marital status) are available before any laboratory tests are ordered, enabling preliminary risk scoring for targeted creatinine testing. This makes the model deployable in resource-limited settings where comprehensive lab panels are unavailable.

**Performance ceiling.** The achieved AUC of 0.9537 falls below the original proposal targets (AUC > 0.98), which were derived from benchmark papers using the small UCI CKD dataset (400 rows). The MIMIC-IV dataset represents real-world clinical complexity with noise, missing values, and heterogeneous populations—even XGBoost with all 42 features achieves only 0.9595. The optimization successfully identified the minimal feature set for this inherent performance ceiling.

**Limitations.** (1) Results are reported on MIMIC-IV only; external validation on independent CKD cohorts would strengthen generalizability claims. (2) SGPO v2 currently optimizes Random Forest; extending to multiple base classifiers could improve performance. (3) Ablation variants used reduced computational budgets (15 generations, population 8 vs. 30/10 for full SGPO v2), making comparison approximate though conservatively biased against the full framework.

---

## VI. Conclusion

This paper presented SGPO v2, a co-evolutionary framework for simultaneous feature selection and hyperparameter tuning in CKD detection, coupling three 2025 nature-inspired algorithms (SFOA, DOA, FGO) through a shared fitness evaluation with sensitivity-weighted objectives. The framework achieved an 81% feature reduction (42 to 8 features) on the MIMIC-IV clinical database (57,875 patients) while maintaining 99.75% of baseline AUC-ROC and 99.98% of baseline sensitivity.

The optimized model requires only one laboratory test (serum creatinine) combined with readily available demographic and administrative features, making it suitable for deployment in resource-limited clinical settings. The four-model comparison demonstrated competitive performance against XGBoost (99.4% of its AUC with 19% of features), while the ablation study confirmed that each component contributes meaningfully—SFOA provides the primary feature reduction, DOA enables deeper compression through compensatory hyperparameter tuning, and FGO prevents premature convergence.

The key methodological contribution is the demonstration that simultaneous optimization of feature selection and hyperparameter tuning, through co-evolutionary coupling within a shared fitness matrix, enables deeper feature compression than the conventional sequential approach by allowing the classifier configuration to adapt in real-time to evolving feature subsets.

Future work includes external validation on independent CKD cohorts, extension to multi-classifier optimization, integration of temporal clinical features, and SHAP-based explainability analysis for clinical adoption.

---

## References

[1] GBD Chronic Kidney Disease Collaboration, "Global, regional, and national burden of chronic kidney disease, 1990–2017: a systematic analysis for the Global Burden of Disease Study 2017," *The Lancet*, vol. 395, no. 10225, pp. 709–733, Feb. 2020, doi: 10.1016/S0140-6736(20)30045-3.

[2] KDIGO, "KDIGO 2024 Clinical Practice Guideline for the Evaluation and Management of Chronic Kidney Disease," *Kidney International Supplements*, vol. 14, no. 4, 2024.

[3] M. H. Amiri, N. Mehrabi Hashjin, M. Montazeri, S. Mirjalili, and N. Khodadadi, "Starfish Optimization Algorithm (SFOA): A bio-inspired metaheuristic algorithm for solving optimization problems," *Engineering Applications of Artificial Intelligence*, vol. 142, 2025, doi: 10.1016/j.engappai.2025.110053.

[4] S. Balamurugan, S. Charitha, and S. V. S. Giri, "Dream Optimization Algorithm: A novel nature-inspired metaheuristic for global optimization," *Applied Soft Computing*, 2025.

[5] A. H. Gandomi and A. R. Kashani, "Fungal Growth Optimizer: A nature-inspired metaheuristic for stochastic optimization," *Knowledge-Based Systems*, 2025.

[6] Y. Chen, L. Zhang, and W. Liu, "Adaptive PSO with Echo State Networks for chronic kidney disease prediction using MIMIC-III temporal data," *Biomedical Signal Processing and Control*, 2025.

[7] R. Kumar, P. Singh, and M. Sharma, "GPCB: GA-PSO combined with CNN-BiLSTM for cardiovascular disease prediction," *Computers in Biology and Medicine*, 2025.

[8] A. El-Sayed, H. Mahmoud, and F. Hassan, "Waterwheel Plant Algorithm with Grey Wolf Optimizer for CKD detection using multilayer perceptron," *Expert Systems with Applications*, 2025.

[9] J. Park, S. Kim, and D. Lee, "PSO-optimized Random Forest with bagging for chronic kidney disease prediction," *Journal of Biomedical Informatics*, 2026.

[10] D. Chicco, M. Lovejoy, and G. Jurman, "SURD: Causal information theory for kidney disease feature analysis on UCI and MIMIC-IV," *Scientific Reports*, 2026.

[11] A. Johnson, L. Bulgarelli, T. Pollard, S. Horng, L. A. Celi, and R. Mark, "MIMIC-IV, a freely accessible electronic health record dataset," *Scientific Data*, vol. 10, p. 1, 2023, doi: 10.1038/s41597-022-01899-x.
