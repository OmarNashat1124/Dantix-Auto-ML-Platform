import logging
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier,
    GradientBoostingRegressor,
    GradientBoostingClassifier,
)
import inspect
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from xgboost import XGBRegressor, XGBClassifier
import xgboost as xgb
from lightgbm import LGBMRegressor, LGBMClassifier
from catboost import CatBoostRegressor, CatBoostClassifier
import os
import joblib
from pipeline.utils_paths import get_user_models

logger = logging.getLogger(__name__)

MODEL_CONFIG_REGISTRY = {
    # ---------------- REGRESSION MODELS ----------------
    "linear_regression": {
        "task": "regression",
        "allowed_params": [
            "fit_intercept",
            "copy_X",
            "n_jobs",
            "positive",
        ],
        "default_params": {
            "fit_intercept": True,
            "copy_X": True,
            "n_jobs": None,
            "positive": False,
        },
    },

    "random_forest_regressor": {
        "task": "regression",
        "allowed_params": [
            "n_estimators",
            "criterion",
            "max_depth",
            "min_samples_split",
            "min_samples_leaf",
            "min_weight_fraction_leaf",
            "max_features",
            "max_leaf_nodes",
            "min_impurity_decrease",
            "bootstrap",
            "oob_score",
            "n_jobs",
            "random_state",
            "max_samples",
        ],
        "default_params": {
            "n_estimators": 300,
            "criterion": "squared_error",
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "min_weight_fraction_leaf": 0.0,
            "max_features": "sqrt",
            "max_leaf_nodes": None,
            "min_impurity_decrease": 0.0,
            "bootstrap": True,
            "oob_score": False,
            "n_jobs": -1,
            "random_state": None,
            "max_samples": None,
        },
    },

    "xgboost_regressor": {
        "task": "regression",
        "allowed_params": [
            "n_estimators",
            "max_depth",
            "learning_rate",
            "subsample",
            "colsample_bytree",
            "gamma",
            "min_child_weight",
            "reg_alpha",
            "reg_lambda",
            "objective",
            "tree_method",
            "n_jobs",
            "early_stopping_rounds",
            "eval_metric",
        ],
        "default_params": {
            "n_estimators": 500,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "gamma": 0.0,
            "min_child_weight": 1.0,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
            "objective": "reg:squarederror",
            "tree_method": "auto",
            "n_jobs": -1,
            "early_stopping_rounds": 50,
            "eval_metric": "rmse",
        },
    },

    "lightgbm_regressor": {
        "task": "regression",
        "allowed_params": [
            "n_estimators",
            "num_leaves",
            "learning_rate",
            "subsample",
            "colsample_bytree",
            "min_child_samples",
            "reg_alpha",
            "reg_lambda",
            "max_depth",
            "min_split_gain",
            "bagging_freq",
            "objective",
            "n_jobs",
            "metric",
            "early_stopping_rounds",
        ],
        "default_params": {
            "n_estimators": 500,
            "num_leaves": 31,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_samples": 20,
            "reg_alpha": 0.0,
            "reg_lambda": 0.0,
            "max_depth": -1,
            "min_split_gain": 0.0,
            "bagging_freq": 0,
            "objective": "regression",
            "n_jobs": -1,
            "metric": "rmse",
            "early_stopping_rounds": 50,
        },
    },

    "gradient_boost_regressor": {
        "task": "regression",
        "allowed_params": [
            "loss",
            "learning_rate",
            "n_estimators",
            "subsample",
            "criterion",
            "min_samples_split",
            "min_samples_leaf",
            "min_weight_fraction_leaf",
            "max_depth",
            "min_impurity_decrease",
            "max_features",
            "alpha",
            "n_iter_no_change",
            "validation_fraction",
        ],
        "default_params": {
            "loss": "squared_error",
            "learning_rate": 0.05,
            "n_estimators": 300,
            "subsample": 1.0,
            "criterion": "friedman_mse",
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "min_weight_fraction_leaf": 0.0,
            "max_depth": 3,
            "min_impurity_decrease": 0.0,
            "max_features": None,
            "alpha": 0.9,
            "n_iter_no_change": None,
            "validation_fraction": 0.1,
        },
    },

    "catboost_regressor": {
        "task": "regression",
        "allowed_params": [
            "iterations",
            "depth",
            "learning_rate",
            "l2_leaf_reg",
            "rsm",
            "loss_function",
            "border_count",
            "bagging_temperature",
            "random_strength",
            "grow_policy",
            "eval_metric",
            "od_type",
            "od_wait",
            "early_stopping_rounds",
        ],
        "default_params": {
            "iterations": 500,
            "depth": 6,
            "learning_rate": 0.05,
            "l2_leaf_reg": 3.0,
            "rsm": 1.0,
            "loss_function": "RMSE",
            "border_count": 254,
            "bagging_temperature": 1.0,
            "random_strength": 1.0,
            "grow_policy": "SymmetricTree",
            "eval_metric": "RMSE",
            "od_type": "Iter",
            "od_wait": 50,
            "early_stopping_rounds": 50,
        },
    },

    "knn_regressor": {
        "task": "regression",
        "allowed_params": [
            "n_neighbors",
            "weights",
            "algorithm",
            "leaf_size",
            "p",
            "metric",
            "n_jobs",
        ],
        "default_params": {
            "n_neighbors": 5,
            "weights": "distance",
            "algorithm": "auto",
            "leaf_size": 30,
            "p": 2,
            "metric": "minkowski",
            "n_jobs": -1,
        },
    },

    # ---------------- CLASSIFICATION MODELS ----------------
    "logistic_regression": {
        "task": "classification",
        "allowed_params": [
            "C",
            "penalty",
            "solver",
            "max_iter",
            "fit_intercept",
            "class_weight",
            "n_jobs",
            "l1_ratio",
            "multi_class",
        ],
        "default_params": {
            "C": 1.0,
            "penalty": "l2",
            "solver": "lbfgs",
            "max_iter": 2000,
            "fit_intercept": True,
            "class_weight": None,
            "n_jobs": -1,
            "l1_ratio": None,
            "multi_class": "auto",
        },
    },

    "random_forest_classifier": {
        "task": "classification",
        "allowed_params": [
            "n_estimators",
            "criterion",
            "max_depth",
            "min_samples_split",
            "min_samples_leaf",
            "min_weight_fraction_leaf",
            "max_features",
            "max_leaf_nodes",
            "min_impurity_decrease",
            "bootstrap",
            "oob_score",
            "n_jobs",
            "random_state",
            "max_samples",
            "class_weight",
        ],
        "default_params": {
            "n_estimators": 300,
            "criterion": "gini",
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "min_weight_fraction_leaf": 0.0,
            "max_features": "sqrt",
            "max_leaf_nodes": None,
            "min_impurity_decrease": 0.0,
            "bootstrap": True,
            "oob_score": False,
            "n_jobs": -1,
            "random_state": None,
            "max_samples": None,
            "class_weight": None,
        },
    },

    "xgboost_classifier": {
        "task": "classification",
        "allowed_params": [
            "n_estimators",
            "max_depth",
            "learning_rate",
            "subsample",
            "colsample_bytree",
            "gamma",
            "min_child_weight",
            "reg_alpha",
            "reg_lambda",
            "objective",
            "tree_method",
            "n_jobs",
            "scale_pos_weight",
            "eval_metric",
            "early_stopping_rounds",
        ],
        "default_params": {
            "n_estimators": 500,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "gamma": 0.0,
            "min_child_weight": 1.0,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
            "objective": "binary:logistic",
            "tree_method": "auto",
            "n_jobs": -1,
            "scale_pos_weight": 1.0,
            "eval_metric": "logloss",
            "early_stopping_rounds": 50,
        },
    },

    "lightgbm_classifier": {
        "task": "classification",
        "allowed_params": [
            "n_estimators",
            "num_leaves",
            "learning_rate",
            "subsample",
            "colsample_bytree",
            "min_child_samples",
            "reg_alpha",
            "reg_lambda",
            "max_depth",
            "min_split_gain",
            "bagging_freq",
            "objective",
            "n_jobs",
            "metric",
            "class_weight",
            "early_stopping_rounds",
        ],
        "default_params": {
            "n_estimators": 500,
            "num_leaves": 31,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_samples": 20,
            "reg_alpha": 0.0,
            "reg_lambda": 0.0,
            "max_depth": -1,
            "min_split_gain": 0.0,
            "bagging_freq": 0,
            "objective": "binary",
            "n_jobs": -1,
            "metric": "logloss",
            "class_weight": None,
            "early_stopping_rounds": 50,
        },
    },

    "gradient_boost_classifier": {
        "task": "classification",
        "allowed_params": [
            "loss",
            "learning_rate",
            "n_estimators",
            "subsample",
            "criterion",
            "min_samples_split",
            "min_samples_leaf",
            "min_weight_fraction_leaf",
            "max_depth",
            "min_impurity_decrease",
            "max_features",
            "n_iter_no_change",
            "validation_fraction",
        ],
        "default_params": {
            "loss": "log_loss",
            "learning_rate": 0.05,
            "n_estimators": 300,
            "subsample": 1.0,
            "criterion": "friedman_mse",
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "min_weight_fraction_leaf": 0.0,
            "max_depth": 3,
            "min_impurity_decrease": 0.0,
            "max_features": None,
            "n_iter_no_change": None,
            "validation_fraction": 0.1,
        },
    },

    "catboost_classifier": {
        "task": "classification",
        "allowed_params": [
            "iterations",
            "depth",
            "learning_rate",
            "l2_leaf_reg",
            "rsm",
            "loss_function",
            "border_count",
            "bagging_temperature",
            "random_strength",
            "grow_policy",
            "eval_metric",
            "scale_pos_weight",
            "od_type",
            "od_wait",
            "early_stopping_rounds",
        ],
        "default_params": {
            "iterations": 500,
            "depth": 6,
            "learning_rate": 0.05,
            "l2_leaf_reg": 3.0,
            "rsm": 1.0,
            "loss_function": "Logloss",
            "border_count": 254,
            "bagging_temperature": 1.0,
            "random_strength": 1.0,
            "grow_policy": "SymmetricTree",
            "eval_metric": "Logloss",
            "scale_pos_weight": 1.0,
            "od_type": "Iter",
            "od_wait": 50,
            "early_stopping_rounds": 50,
        },
    },

    "knn_classifier": {
        "task": "classification",
        "allowed_params": [
            "n_neighbors",
            "weights",
            "algorithm",
            "leaf_size",
            "p",
            "metric",
            "n_jobs",
        ],
        "default_params": {
            "n_neighbors": 5,
            "weights": "distance",
            "algorithm": "auto",
            "leaf_size": 30,
            "p": 2,
            "metric": "minkowski",
            "n_jobs": -1,
        },
    },
}


def build_model_list_from_config(model_config: dict, task_type: str) -> list[dict]:
    """
    Convert an agent-provided config into the `model_list` structure that
    model_training_agent expects.

    model_config schema (from the agent):

    {
      "models": [
        {
          "model_name": "lightgbm_regressor",
          "enabled": true,
          "params": { ... },
          "experiment_name": "optional_experiment_name"
        },
        ...
      ]
    }
    """
    models = model_config.get("models", []) or []
    model_list: list[dict] = []

    for m in models:
        name = m.get("model_name")
        if not name:
            continue

        if not m.get("enabled", True):
            continue

        schema = MODEL_CONFIG_REGISTRY.get(name)
        if schema is None:
            continue  # unknown model

        if schema["task"] != task_type:
            continue  # wrong task (reg vs clf)

        # start from defaults
        params = dict(schema.get("default_params", {}))

        # override only allowed params
        raw_params = m.get("params", {}) or {}
        for k, v in raw_params.items():
            if k in schema["allowed_params"]:
                params[k] = v

        model_list.append(
            {
                "model_name": name,
                "params": params,
                "experiment_name": m.get("experiment_name"),
            }
        )

    return model_list


def agentic_train_test_split(
    df,
    target_column,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = False,
    is_classification: bool = False,
):
    """
    Split data into train/test with some extra safety:

    - Step 5: Drop rows where the target is NaN.
    - Step 3: Use stratified split for classification *when possible*.
    - Step 4: Guard against single-class training sets for classification.
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame.")

    # --- Step 5: Explicitly handle NaNs in target ---
    before_rows = len(df)
    df = df[df[target_column].notna()].copy()

    if df.empty:
        raise ValueError(
            f"All rows were dropped because '{target_column}' is NaN. "
            "Cannot proceed with train/test split."
        )

    dropped = before_rows - len(df)
    if dropped > 0:
        logger.info(
            "Dropped %d rows with NaN in target column '%s' before train/test split.",
            dropped,
            target_column,
        )

    X = df.drop(columns=[target_column])
    y = df[target_column]

    # --- Step 3: Stratified split for classification (when feasible) ---
    strat_labels = None
    if stratify and is_classification:
        vc = y.value_counts()
        n_classes = vc.size

        if n_classes < 2:
 
            raise ValueError(
                f"Classification task but target '{target_column}' has only {n_classes} class "
                f"after cleaning: {vc.index.tolist()}."
            )


        if (vc < 2).any():
            logger.warning(
                "Some classes in target '%s' have fewer than 2 samples %s. "
                "Disabling stratified split to avoid train_test_split error.",
                target_column,
                vc.to_dict(),
            )
            strat_labels = None
        else:
            strat_labels = y
    else:
        strat_labels = None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=strat_labels,
    )

    if is_classification:
        n_train_classes = y_train.nunique()
        if n_train_classes < 2:
            raise ValueError(
                "After train/test split, y_train for classification has only "
                f"{n_train_classes} class. This usually means the dataset is very small, "
                "extremely imbalanced, or too many rows were removed during cleaning/outlier handling. "
                "Try adjusting test_size or the cleaning thresholds."
            )

    return X_train, X_test, y_train, y_test


def regression_metrics(y_true, y_pred):
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def classification_metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def train_model(
    ModelClass,
    X_train,
    y_train,
    X_test,
    y_test,
    params,
    metric_fn,
    model_name: str,
    experiment_name: str | None,  # kept for compatibility, unused
):
    """
    Generic model training helper (no MLflow).

    - Computes train/test metrics.
    - Supports optional early stopping for XGBoost / LightGBM / CatBoost.
    - Uses NumPy arrays for XGBoost to avoid pandas dtype issues.
    - Handles multi-class classification correctly for XGBoost, LightGBM, CatBoost.
    """
    from xgboost import XGBRegressor, XGBClassifier

    try:
        from lightgbm import LGBMRegressor, LGBMClassifier
    except ImportError:
        LGBMRegressor = LGBMClassifier = None

    try:
        from catboost import CatBoostRegressor, CatBoostClassifier
    except ImportError:
        CatBoostRegressor = CatBoostClassifier = None

    # Copy params so we can safely modify them
    params = dict(params or {})

    # Extract early_stopping_rounds so it doesn't go into the constructor
    early_stopping_rounds = params.pop("early_stopping_rounds", None)

    # ------------------------------------------------------------------
    # Create model instance
    # ------------------------------------------------------------------
    try:
        model = ModelClass(**params)
    except TypeError as e:
        err = str(e)
        logger.error(f"Error creating model {ModelClass.__name__}: {err}")
        return {"error": err, "params": params}

    # ------------------------------------------------------------------
    # Detect classification cardinality for classifiers
    # ------------------------------------------------------------------
    n_classes = None
    classes = None

    is_xgb_clf = isinstance(model, XGBClassifier)
    is_lgbm_clf = LGBMClassifier is not None and isinstance(model, LGBMClassifier)
    is_cat_clf = CatBoostClassifier is not None and isinstance(model, CatBoostClassifier)

    if is_xgb_clf or is_lgbm_clf or is_cat_clf:
        classes = np.unique(y_train)
        n_classes = int(len(classes))

    # ------------------------------------------------------------------
    # Clean up scale_pos_weight for multi-class boosters
    # ------------------------------------------------------------------
    if n_classes is not None and n_classes > 2 and "scale_pos_weight" in params:
        logger.warning(
            "Removing 'scale_pos_weight' from params for multi-class (n_classes=%d). "
            "It is only meaningful/allowed for binary classification in some libraries.",
            n_classes,
        )
        params.pop("scale_pos_weight", None)
        # Recreate the model without scale_pos_weight
        model = ModelClass(**params)
        # Recompute type flags (class stays the same, but be explicit)
        is_xgb_clf = isinstance(model, XGBClassifier)
        is_lgbm_clf = LGBMClassifier is not None and isinstance(model, LGBMClassifier)
        is_cat_clf = CatBoostClassifier is not None and isinstance(model, CatBoostClassifier)

    # ------------------------------------------------------------------
    # Configure XGBoost classifier for binary vs multi-class
    # ------------------------------------------------------------------
    if is_xgb_clf and n_classes is not None:
        if n_classes > 2:
            # Multi-class
            model.set_params(objective="multi:softprob", num_class=n_classes)
            eval_metric = model.get_params().get("eval_metric", None)
            if eval_metric is None or eval_metric in ("logloss", "error", "auc", "aucpr"):
                model.set_params(eval_metric="mlogloss")
        else:
            # Binary
            model.set_params(objective="binary:logistic")

    # ------------------------------------------------------------------
    # Configure LightGBM classifier for binary vs multi-class
    # ------------------------------------------------------------------
    if is_lgbm_clf and n_classes is not None:
        if n_classes > 2:
            # Multi-class
            model.set_params(objective="multiclass", num_class=n_classes)
            params_now = model.get_params()
            metric = params_now.get("metric", None)
            if metric is None or metric in ("logloss", "binary_logloss", "auc"):
                model.set_params(metric="multi_logloss")
        else:
            # Binary
            model.set_params(objective="binary")

    # ------------------------------------------------------------------
    # Configure CatBoost classifier for binary vs multi-class
    # ------------------------------------------------------------------
    if is_cat_clf and n_classes is not None:
        if n_classes > 2:
            # Multi-class: force BOTH loss_function and eval_metric to MultiClass
            model.set_params(loss_function="MultiClass", eval_metric="MultiClass")
        else:
            # Binary: force loss_function / eval_metric to Logloss unless user overrides later
            params_now = model.get_params()
            current_loss = params_now.get("loss_function", None)
            current_metric = params_now.get("eval_metric", None)

            if current_loss is None or current_loss.lower() in ("multiclass", "multiclassova", "multi_label"):
                model.set_params(loss_function="Logloss")
            if current_metric is None or current_metric.lower() in ("multiclass", "multiclassova", "multi_label"):
                model.set_params(eval_metric="Logloss")

    # Final params (for logging / debugging)
    try:
        final_params = model.get_params(deep=False)
    except TypeError:
        final_params = model.get_params()

    # ------------------------------------------------------------------
    # Build fit kwargs (early stopping / eval_set)
    # ------------------------------------------------------------------
    Xtr, Xte, ytr, yte = X_train, X_test, y_train, y_test
    fit_kwargs: dict = {}

    # XGBoost → use numpy + eval_set + early_stopping_rounds
    # (classic API, works on older XGBoost versions under Python 3.10.18)
    if isinstance(model, (XGBRegressor, XGBClassifier)):
        Xtr = np.asarray(X_train)
        Xte = np.asarray(X_test)
        ytr = np.asarray(y_train)
        yte = np.asarray(y_test)

        fit_kwargs["eval_set"] = [(Xtr, ytr), (Xte, yte)]
        fit_kwargs.setdefault("verbose", False)

        if early_stopping_rounds is not None:
            fit_kwargs["early_stopping_rounds"] = early_stopping_rounds


    elif LGBMRegressor is not None and isinstance(
        model, (LGBMRegressor, LGBMClassifier)
    ):

        fit_kwargs["eval_set"] = [(X_test, y_test)]

        if early_stopping_rounds is not None:
            if "eval_metric" not in fit_kwargs:
                if isinstance(model, LGBMRegressor):

                    fit_kwargs["eval_metric"] = "rmse"
                else:

                    fit_kwargs["eval_metric"] = "logloss"

            # 2) Prefer callback-based early stopping (LightGBM ≥ 4)
            try:
                # local import, does NOT change the global import section
                import lightgbm as _lgb

                callbacks = list(fit_kwargs.get("callbacks", []))
                callbacks.append(
                    _lgb.early_stopping(
                        stopping_rounds=early_stopping_rounds,
                        verbose=False,
                    )
                )
                fit_kwargs["callbacks"] = callbacks
            except Exception:
                # Fallback for older versions that still accept the kwarg
                fit_kwargs["early_stopping_rounds"] = early_stopping_rounds

        # Some LightGBM sklearn wrappers don't accept `verbose` in .fit()
        # Remove it here so the generic fit() call below doesn't crash.
        fit_kwargs.pop("verbose", None)



    # CatBoost → eval_set; early_stopping_rounds allowed via param
    elif CatBoostRegressor is not None and isinstance(
        model, (CatBoostRegressor, CatBoostClassifier)
    ):
        fit_kwargs["eval_set"] = [(X_test, y_test)]
        if early_stopping_rounds is not None:
            fit_kwargs["early_stopping_rounds"] = early_stopping_rounds

    # ------------------------------------------------------------------
    # Fit with safe fallback when early_stopping_rounds is unsupported
    # ------------------------------------------------------------------
    try:
        model.fit(Xtr, ytr, **fit_kwargs)
    except TypeError as e:
        msg = str(e)

            # ------------------------------------------------------------------
    # Build fit kwargs (early stopping / eval_set)
    # ------------------------------------------------------------------
    Xtr, Xte, ytr, yte = X_train, X_test, y_train, y_test
    fit_kwargs: dict = {}

    # XGBoost → use numpy + eval_set + early_stopping_rounds
    if isinstance(model, (XGBRegressor, XGBClassifier)):
        Xtr = np.asarray(X_train)
        Xte = np.asarray(X_test)
        ytr = np.asarray(y_train)
        yte = np.asarray(y_test)

        fit_kwargs["eval_set"] = [(Xtr, ytr), (Xte, yte)]
        fit_kwargs.setdefault("verbose", False)

        if early_stopping_rounds is not None:
            fit_kwargs["early_stopping_rounds"] = early_stopping_rounds

    elif LGBMRegressor is not None and isinstance(
        model, (LGBMRegressor, LGBMClassifier)
    ):
        fit_kwargs["eval_set"] = [(X_test, y_test)]
        if early_stopping_rounds is not None:
            fit_kwargs["early_stopping_rounds"] = early_stopping_rounds


        else:
            raise


    # ------------------------------------------------------------------
    # Predictions & metrics
    # ------------------------------------------------------------------
    if isinstance(model, (XGBRegressor, XGBClassifier)):
        train_preds = model.predict(Xtr)
        test_preds = model.predict(Xte)
    else:
        train_preds = model.predict(X_train)
        test_preds = model.predict(X_test)

    train_m = metric_fn(y_train, train_preds)
    test_m = metric_fn(y_test, test_preds)

    logger.info(
        "%s | params=%s | train_metrics=%s | test_metrics=%s",
        model_name,
        final_params,
        train_m,
        test_m,
    )

    return {
        "model": model,
        "model_name": model_name,
        "train_metrics": train_m,
        "test_metrics": test_m,
        "params": final_params,
    }




# -------------------------------------------------------------------
# TRAINER WRAPPERS (same structure as before)
# -------------------------------------------------------------------

def train_linear_regression(
    X_train, y_train, X_test, y_test, params, experiment_name
):
    return train_model(
        LinearRegression,
        X_train,
        y_train,
        X_test,
        y_test,
        params,
        regression_metrics,
        "linear_regression",
        experiment_name,
    )


def train_random_forest_regressor(
    X_train, y_train, X_test, y_test, params, experiment_name
):
    return train_model(
        RandomForestRegressor,
        X_train,
        y_train,
        X_test,
        y_test,
        params,
        regression_metrics,
        "random_forest_regressor",
        experiment_name,
    )


def train_xgboost_regressor(
    X_train, y_train, X_test, y_test, params, experiment_name
):
    return train_model(
        XGBRegressor,
        X_train,
        y_train,
        X_test,
        y_test,
        params,
        regression_metrics,
        "xgboost_regressor",
        experiment_name,
    )


def train_lightgbm_regressor(
    X_train, y_train, X_test, y_test, params, experiment_name
):
    return train_model(
        LGBMRegressor,
        X_train,
        y_train,
        X_test,
        y_test,
        params,
        regression_metrics,
        "lightgbm_regressor",
        experiment_name,
    )


def train_gradient_boost_regressor(
    X_train, y_train, X_test, y_test, params, experiment_name
):
    return train_model(
        GradientBoostingRegressor,
        X_train,
        y_train,
        X_test,
        y_test,
        params,
        regression_metrics,
        "gradient_boost_regressor",
        experiment_name,
    )


def train_catboost_regressor(
    X_train, y_train, X_test, y_test, params, experiment_name
):
    return train_model(
        CatBoostRegressor,
        X_train,
        y_train,
        X_test,
        y_test,
        params,
        regression_metrics,
        "catboost_regressor",
        experiment_name,
    )


def train_knn_regressor(X_train, y_train, X_test, y_test, params, experiment_name):
    return train_model(
        KNeighborsRegressor,
        X_train,
        y_train,
        X_test,
        y_test,
        params,
        regression_metrics,
        "knn_regressor",
        experiment_name,
    )


def train_logistic_regression(
    X_train, y_train, X_test, y_test, params, experiment_name
):
    return train_model(
        LogisticRegression,
        X_train,
        y_train,
        X_test,
        y_test,
        params,
        classification_metrics,
        "logistic_regression",
        experiment_name,
    )


def train_random_forest_classifier(
    X_train, y_train, X_test, y_test, params, experiment_name
):
    return train_model(
        RandomForestClassifier,
        X_train,
        y_train,
        X_test,
        y_test,
        params,
        classification_metrics,
        "random_forest_classifier",
        experiment_name,
    )


def train_xgboost_classifier(
    X_train, y_train, X_test, y_test, params, experiment_name
):
    return train_model(
        XGBClassifier,
        X_train,
        y_train,
        X_test,
        y_test,
        params,
        classification_metrics,
        "xgboost_classifier",
        experiment_name,
    )


def train_lightgbm_classifier(
    X_train, y_train, X_test, y_test, params, experiment_name
):
    return train_model(
        LGBMClassifier,
        X_train,
        y_train,
        X_test,
        y_test,
        params,
        classification_metrics,
        "lightgbm_classifier",
        experiment_name,
    )


def train_gradient_boost_classifier(
    X_train, y_train, X_test, y_test, params, experiment_name
):
    return train_model(
        GradientBoostingClassifier,
        X_train,
        y_train,
        X_test,
        y_test,
        params,
        classification_metrics,
        "gradient_boost_classifier",
        experiment_name,
    )


def train_catboost_classifier(
    X_train, y_train, X_test, y_test, params, experiment_name
):
    return train_model(
        CatBoostClassifier,
        X_train,
        y_train,
        X_test,
        y_test,
        params,
        classification_metrics,
        "catboost_classifier",
        experiment_name,
    )


def train_knn_classifier(X_train, y_train, X_test, y_test, params, experiment_name):
    return train_model(
        KNeighborsClassifier,
        X_train,
        y_train,
        X_test,
        y_test,
        params,
        classification_metrics,
        "knn_classifier",
        experiment_name,
    )


def save_best_model(user_id: str, model, model_name: str):
    folder = get_user_models(user_id)
    path = os.path.join(folder, f"{model_name}.pkl")
    joblib.dump(model, path)
    return path


def select_best_model(model_results, task_type: str):
    """
    Select the best model and return its metadata.
    Saving will be handled in modal_app.py via a Volume.
    """
    if task_type == "regression":
        best = min(model_results, key=lambda x: x["test_metrics"]["rmse"])
    else:
        best = max(model_results, key=lambda x: x["test_metrics"]["accuracy"])

    return {
        "best_model_name": best["model_name"],
        "best_model": best["model"],
        "train_metrics": best["train_metrics"],
        "test_metrics": best["test_metrics"],
        "params": best["params"],
    }

