# Agreement Without Faithfulness: Do Consistent XAI Methods Produce Better Explanations?

> **Research Question:** Does pairwise agreement between Explainable AI (XAI) feature attribution methods serve as a reliable indicator of explanation faithfulness in tabular machine learning?

This repository contains the complete experimental code, raw and processed data artifacts, and manuscript for an empirical study evaluating the relationship between inter-method agreement and explanation faithfulness.

**Publication Status:** Not submitted for publication (independent research manuscript).


## Abstract

Practitioners of Explainable AI (XAI) often validate feature attributions by comparing multiple methods: if SHAP and LIME identify similar features as influential, the resulting explanation is intuitively assumed to be trustworthy. This study empirically examines whether inter-method agreement is an informative proxy for explanation quality.

Using an XGBoost classifier trained on the UCI Adult Income benchmark, we generate instance-level explanations via TreeSHAP, LIME, and Permutation Sensitivity, and evaluate their pairwise agreement (Spearman rank correlation $\rho$, Kendall $\tau$, and top-$k$ overlap), faithfulness (comprehensiveness and sufficiency under marginal imputation), and local stability under continuous feature perturbation.

Our results demonstrate that:
1. Explanation methods exhibit moderate, highly variable agreement ($\rho = 0.431$ to $0.749$), with frequent per-instance contradictions.
2. Comprehensiveness and sufficiency yield consistent aggregate rankings but diverge at the instance level for 10.4% to 18.6% of test instances.
3. Inter-method agreement shows negligible-to-small negative correlations with faithfulness (Spearman $\rho = -0.133$ to $0.004$); notably, agreement between LIME and Permutation Sensitivity correlates negatively with comprehensiveness ($p = 0.003$, Holm-adjusted $p = 0.009$).
4. TreeSHAP and LIME demonstrate comparable rank stability under 1% input perturbations ($p = 0.297$), and LIME exhibits minimal internal Monte Carlo sampling stochasticity across random seeds ($\rho = 0.925 \pm 0.014$).

These findings suggest that agreement between explanation methods is not, by itself, a sufficient validation signal for explanation faithfulness.


## Key Findings

### RQ1: Inter-Method Agreement
Evaluated across $N = 500$ stratified test instances on the UCI Adult Income dataset:
- **SHAP vs. Permutation Sensitivity:** Spearman $\rho = 0.749 \pm 0.118$, Kendall $\tau = 0.592 \pm 0.118$, Top-3 overlap = $0.649 \pm 0.208$, Top-5 overlap = $0.741 \pm 0.147$.
- **SHAP vs. LIME:** Spearman $\rho = 0.574 \pm 0.155$, Kendall $\tau = 0.418 \pm 0.129$, Top-3 overlap = $0.357 \pm 0.224$, Top-5 overlap = $0.604 \pm 0.153$.
- **LIME vs. Permutation Sensitivity:** Spearman $\rho = 0.431 \pm 0.228$, Kendall $\tau = 0.318 \pm 0.176$, Top-3 overlap = $0.404 \pm 0.243$, Top-5 overlap = $0.585 \pm 0.169$.

Methods agree moderately on average, but individual instances exhibit substantial dispersion and occasional complete disagreement.

### RQ2: Faithfulness and Metric Consistency
Evaluated via Comprehensiveness (higher is better) and Sufficiency (lower is better):
- **Comprehensiveness ($k=3$):** TreeSHAP ($0.2269 \pm 0.1844$) > Permutation Sensitivity ($0.2042 \pm 0.2049$) > LIME ($0.0699 \pm 0.1549$) > Random ($0.0367 \pm 0.1019$).
- **Sufficiency ($k=3$):** TreeSHAP ($0.0399 \pm 0.0939$) < Permutation Sensitivity ($0.0585 \pm 0.1074$) < LIME ($0.1810 \pm 0.1865$) < Random ($0.2780 \pm 0.2034$).
- **At $k=1$ Sufficiency:** Random baseline ($0.3740 \pm 0.1941$) outperforms LIME ($0.3928 \pm 0.2180$).

Comprehensiveness and sufficiency agree in aggregate at $k \ge 3$, but disagree on which method produced the best explanation for 10.4% to 18.6% of individual instances.

### RQ3: Agreement vs. Faithfulness
Correlation between pairwise explanation agreement ($\rho$) and pair-averaged comprehensiveness ($k=3$):
- **SHAP vs. LIME:** Pearson $r = -0.045$ ($p = 0.312$); Spearman $\rho = -0.083$ (raw $p = 0.064$, Holm-adjusted $p = 0.129$).
- **SHAP vs. PS:** Pearson $r = -0.017$ ($p = 0.710$); Spearman $\rho = 0.004$ (raw $p = 0.921$, Holm-adjusted $p = 0.921$).
- **LIME vs. PS:** Pearson $r = -0.034$ ($p = 0.449$); Spearman $\rho = -0.133$ (raw $p = 0.003$, Holm-adjusted $p = 0.009$).
- **Sensitivity Analysis:** Correlating agreement directly against each method's individual faithfulness score yields weak correlations ($|r| \le 0.171$) without joint positive alignment, confirming that the lack of positive relationship is not an artifact of pair-averaging.

### RQ4: Attribution Stability
- **1% Gaussian Input Perturbation:** TreeSHAP rank stability $\rho = 0.876 \pm 0.062$ vs. LIME $\rho = 0.886 \pm 0.046$ (Wilcoxon $W = 2176, p = 0.297$, matched-pairs rank-biserial $r_{\text{rb}} = -0.121$).
- **LIME Seed Sensitivity Control:** Holding inputs constant and varying the Monte Carlo sampling seed across 10 trials yields mean pairwise $\rho = 0.925 \pm 0.014$ (median $0.926$), showing minimal sampling noise at 5,000 samples.

 

## Methodology

### Dataset and Model
- **Dataset:** UCI Adult Income (OpenML ID 1590, 45,222 instances, 14 features). Stratified 80/20 train/test split.
- **Model:** XGBoost Classifier (300 estimators, max depth 6, learning rate 0.1, subsample 0.8, colsample 0.8). Test accuracy = 86.76%, F1 = 0.7122, AUC = 0.9250.
- **Evaluation Sample:** $N = 500$ stratified instances across prediction confidence terciles and predicted classes.

### Evaluated Methods
1. **TreeSHAP:** Exact tree-path polynomial Shapley values computed on model margin output.
2. **LIME:** Local linear surrogate fitted with 5,000 perturbation samples per instance in probability space.
3. **Permutation Sensitivity (PS):** Instance-level probability sensitivity under feature permutation across evaluation samples (20 repeats).
4. **Random Baseline:** Uniform random attribution vector serving as a null control.

### Faithfulness Formulations
- **Comprehensiveness ($k$):** Measures the drop in predicted probability for the target class upon removing the top-$k$ most important features (imputed using training set median/mode):
  $$\text{Comp}(x, k) = f(x)_c - f(x_{\setminus \text{top-}k})_c$$
- **Sufficiency ($k$):** Measures prediction retention when keeping only the top-$k$ features:
  $$\text{Suff}(x, k) = f(x)_c - f(x_{\text{only top-}k})_c$$

 

## Reproduction

### Environment Setup
Python 3.10+ is required (tested on Python 3.12.3).

```bash
git clone https://github.com/Angat-Shah/xai-agreement-without-faithfulness.git
cd xai-agreement-without-faithfulness
pip install -r requirements.txt
```

Alternatively, to install the locked environment:

```bash
pip install -r requirements-lock.txt
```

### Running Experiments
To execute the end-to-end pipeline (dataset loading, model training, explanation generation, agreement, faithfulness, and stability evaluations):

```bash
python3 src/run_experiments.py
```

Outputs are saved in `results/raw/` and `results/processed/`.

### Generating Figures and Tables
To regenerate all vector PDF and PNG figures in `results/figures/` and LaTeX tables in `results/tables/`:

```bash
python3 src/generate_figures.py
```

### Compiling the Manuscript
The LaTeX manuscript in `paper/` references the generated figures and tables in `results/`:

```bash
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
cp main.pdf manuscript.pdf
```

 

## Repository Structure

```
xai-agreement-without-faithfulness/
  README.md                    Primary overview and reproduction instructions
  LICENSE                      MIT License
  CITATION.cff                 Machine-readable citation metadata
  .gitignore                   Git exclusion rules
  requirements.txt             Core experiment dependencies
  requirements-lock.txt        Full environment dependency lock
  src/
    __init__.py
    config.py                  Experiment parameters and directory paths
    run_experiments.py         End-to-end experiment pipeline
    generate_figures.py        Script producing figures and LaTeX tables
  results/
    raw/                       Raw explanation matrices and evaluation outputs
      X_sample.csv
      y_sample.csv
      shap_values.npy
      lime_values.npy
      perm_values.npy
      detailed_results.json
    processed/                 Consolidated metrics and summary statistics
      all_results.json
    figures/                   Vector PDF and PNG figures
      fig1_agreement_distributions.pdf
      fig1_agreement_distributions.png
      fig2_topk_agreement.pdf
      fig2_topk_agreement.png
      fig3_faithfulness.pdf
      fig3_faithfulness.png
      fig4_agreement_vs_faithfulness.pdf
      fig4_agreement_vs_faithfulness.png
      fig5_stability.pdf
      fig5_stability.png
      fig6_metric_rankings.pdf
      fig6_metric_rankings.png
    tables/                    Generated LaTeX tables
      table1_model_performance.tex
      table2_agreement.tex
      table3_faithfulness.tex
      table4_agree_vs_faith.tex
      table5_stability.tex
  paper/
    main.tex                   IEEEtran LaTeX manuscript source
    references.bib             Bibliographic references
    manuscript.pdf             Compiled six-page manuscript PDF
```

 

## Manuscript

The full manuscript is available at [`paper/manuscript.pdf`](paper/manuscript.pdf). It is an IEEEtran-formatted six-page conference-style paper with two-column layout, vector figures, and balanced references.

 

## Research Provenance

This research developed through two stages:

1. **Coursework Origin (September 2025):** As part of my seventh-semester Explainable AI coursework, I reviewed XAI evaluation literature and submitted two review papers on selected XAI topics on September 13, 2025.
2. **Independent Empirical Study (2025–2026):** Those reviews led me to question whether agreement between different explanation methods can actually indicate explanation faithfulness. I then independently formulated the research questions and experimental protocol, implemented the evaluation pipeline, ran the experiments, performed the statistical analyses, including Holm correction, sensitivity tests, and seed controls, and wrote the manuscript.

 

## Citation

```bibtex
@misc{shah2026agreement,
  author    = {Angat Shah},
  title     = {Agreement Without Faithfulness: Do Consistent {XAI} Methods Produce Better Explanations?},
  year      = {2026},
  note      = {Independent Research Manuscript; not submitted for publication},
  url       = {https://github.com/Angat-Shah/xai-agreement-without-faithfulness}
}
```

 

## License

This project is licensed under the [MIT License](LICENSE).
