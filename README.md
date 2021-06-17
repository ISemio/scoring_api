
# About model :
LightGBM parameters found by Bayesian optimization ( https://github.com/fmfn/BayesianOptimization )
StratifiedKFold
Combined random oversampling and undersampling for imbalanced data
    Shape data before:  (307511, 122)
    Original data:
    Counter({0: 282686, 1: 24825})
    Oversampled data:
    Counter({0: 282686, 1: 56537})
    Undersampled data:
    Counter({0: 80767, 1: 56537})
    Shape data after:  (137304, 380)

Evaluation metrics: AUC, ROC curve, f1_score, precision, confusion matrix
    AUC = 0.93
    f1_score = 0.88
Data shape randomized for dashboard (25000, 385)