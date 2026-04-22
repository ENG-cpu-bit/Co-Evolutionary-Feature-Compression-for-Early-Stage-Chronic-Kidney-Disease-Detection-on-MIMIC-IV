"""
fitness.py
==========
Shared fitness function for SGPO v2.
Evaluates a (feature_mask, hyperparameters) pair using inner CV.

Fitness = 0.50 * AUC - 0.20 * (n_selected / n_total) + 0.30 * Sensitivity
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from sklearn.experimental import enable_iterative_imputer  # noqa
    from sklearn.impute import IterativeImputer
    HAS_MICE = True
except ImportError:
    HAS_MICE = False

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False


# Fitness weights (from proposal)
W_AUC = 0.50
W_FEATURE_PENALTY = 0.20
W_SENSITIVITY = 0.30


def evaluate_solution(X, y, feature_mask, hp_dict, n_inner_folds=3,
                      use_smote=True, use_mice=False, random_state=42):
    """
    Evaluate a single (feature_mask, hyperparameters) solution.

    Parameters
    ----------
    X : pd.DataFrame — full feature matrix
    y : pd.Series — binary labels
    feature_mask : np.array — binary mask (1=select, 0=skip)
    hp_dict : dict — Random Forest hyperparameters
    n_inner_folds : int — inner CV folds
    use_smote : bool — apply SMOTE on training folds
    use_mice : bool — use MICE imputation (else median)
    random_state : int

    Returns
    -------
    fitness : float — combined fitness score
    auc : float — mean AUC across inner folds
    sensitivity : float — mean sensitivity across inner folds
    n_features : int — number of selected features
    """
    # Select features
    selected_idx = np.where(feature_mask == 1)[0]
    n_selected = len(selected_idx)
    n_total = len(feature_mask)

    # Must have at least 2 features
    if n_selected < 2:
        return -1.0, 0.0, 0.0, n_selected

    X_sub = X.iloc[:, selected_idx].values
    y_arr = y.values

    # Build pipeline
    if use_mice and HAS_MICE:
        imputer = IterativeImputer(max_iter=10, random_state=random_state)
    else:
        imputer = SimpleImputer(strategy="median")

    clf = RandomForestClassifier(
        n_estimators=hp_dict.get("n_estimators", 100),
        max_depth=hp_dict.get("max_depth", None),
        min_samples_split=hp_dict.get("min_samples_split", 2),
        min_samples_leaf=hp_dict.get("min_samples_leaf", 1),
        random_state=random_state,
        n_jobs=-1,
    )

    if use_smote and HAS_SMOTE:
        pipe = ImbPipeline([
            ("imputer", imputer),
            ("scaler", StandardScaler()),
            ("smote", SMOTE(random_state=random_state)),
            ("classifier", clf),
        ])
    else:
        pipe = Pipeline([
            ("imputer", imputer),
            ("scaler", StandardScaler()),
            ("classifier", clf),
        ])

    # Inner cross-validation
    inner_cv = StratifiedKFold(
        n_splits=n_inner_folds, shuffle=True, random_state=random_state
    )

    auc_scores = []
    sens_scores = []

    for train_idx, val_idx in inner_cv.split(X_sub, y_arr):
        X_train, X_val = X_sub[train_idx], X_sub[val_idx]
        y_train, y_val = y_arr[train_idx], y_arr[val_idx]

        try:
            pipe.fit(X_train, y_train)
            y_prob = pipe.predict_proba(X_val)[:, 1]
            y_pred = pipe.predict(X_val)

            auc_scores.append(roc_auc_score(y_val, y_prob))
            sens_scores.append(recall_score(y_val, y_pred))
        except Exception:
            auc_scores.append(0.5)
            sens_scores.append(0.0)

    mean_auc = np.mean(auc_scores)
    mean_sens = np.mean(sens_scores)
    feature_ratio = n_selected / n_total

    fitness = (
        W_AUC * mean_auc
        - W_FEATURE_PENALTY * feature_ratio
        + W_SENSITIVITY * mean_sens
    )

    return fitness, mean_auc, mean_sens, n_selected
