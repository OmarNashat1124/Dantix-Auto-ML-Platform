import os
import json
import shutil
import modal
import numpy as np
import pandas as pd
from modal import Volume, Image ,Secret
from groq import Groq

import logging
import os

# Path to logs folder inside the Modal volume
LOGS_DIR = "/root/data/logs"
os.makedirs(LOGS_DIR, exist_ok=True)

# Create file handler
file_handler = logging.FileHandler(os.path.join(LOGS_DIR, "pipeline.log"))

# Create stream handler (for modal console logs)
stream_handler = logging.StreamHandler()

# Apply config
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[file_handler, stream_handler],
    force=True
)

logger = logging.getLogger(__name__)


volume = Volume.from_name("ai_pipeline_volume", create_if_missing=True)

ai_image = (
    Image.debian_slim()
    .pip_install(
        "imbalanced-learn",
        "pyarrow",
        "pandas",
        "numpy",
        "scikit-learn",
        "xgboost==1.7.6",
        "lightgbm",
        "catboost",
        "mlflow",
        "matplotlib",
        "openpyxl",
        "evidently",
        "optuna",
        "joblib",
        "requests",
        "plotly",
        "dash",
        "groq",
    )
    .add_local_python_source("pipeline")
)

groq_secret = Secret.from_name("groq-api-key")

app = modal.App("agentic-ai-app")

@app.function(image=ai_image, volumes={"/root/data": volume})
def init_volume():
    """
    Initializes the Modal volume structure for the Agentic AI system.
    Creates base directories:
       /root/data/users
       /root/data/logs
       /root/data/tmp (optional)
    This should be run ONCE before any user uploads files.
    """
    base = "/root/data"

    users_dir = os.path.join(base, "users")
    logs_dir = os.path.join(base, "logs")

    os.makedirs(users_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)


    return {
        "status": "success",
        "created": {
            "users_dir": users_dir,
            "logs_dir": logs_dir,
        }
    }
def _detect_latest_version(folder: str, prefix: str, ext: str):
    files = [
        f for f in os.listdir(folder)
        if f.startswith(prefix) and f.endswith(ext)
    ]
    if not files:
        return None
    versions = []
    for f in files:
        try:
            v = int(f.split("_v")[1].split(".")[0])
            versions.append(v)
        except:
            pass
    return max(versions) if versions else None


def _next_version(folder: str, prefix: str, ext: str):
    latest = _detect_latest_version(folder, prefix, ext)
    return 1 if latest is None else latest + 1


@app.function(image=ai_image, volumes={"/root/data": volume})
def ingestion_agent(user_id: str, filename: str, url:str):

    from pipeline.Data_Ingestion import load_file
    from pipeline.utils_paths import get_user_uploads

    uploads_folder = get_user_uploads(user_id)
    df = load_file(user_id, filename, url)

    version = _next_version(uploads_folder, "ingested_", ".parquet")
    out_path = os.path.join(uploads_folder, f"ingested_v{version}.parquet")
    df.to_parquet(out_path, index=False)

    return {
        "status": "success",
        "user_id": user_id,
        "version": version,
        "saved_file": out_path,
        "shape": list(df.shape),
        "columns": df.columns.tolist(),
        "sample_rows": df.head(3).to_dict(orient="records"),
    }



@app.function(image=ai_image, volumes={"/root/data": volume})
def preprocessing_agent(user_id: str, target_column: str):

    from pipeline.Preprocessing import clean_data
    from pipeline.utils_paths import get_user_uploads, get_user_preprocessed

    uploads = get_user_uploads(user_id)
    ingested_files = [
        f for f in os.listdir(uploads)
        if f.startswith("ingested_") and f.endswith(".parquet")
    ]
    if not ingested_files:
        return {"status": "error", "message": "No ingested files found."}

    latest_ingested = max(
        ingested_files,
        key=lambda f: int(f.split("_v")[1].split(".")[0])
    )
    version = int(latest_ingested.split("_v")[1].split(".")[0])

    df = pd.read_parquet(os.path.join(uploads, latest_ingested))

    cleaned_df, num_cont, num_cat, cat_cols = clean_data(df, target_column)

    pre_folder = get_user_preprocessed(user_id)
    cleaned_path = os.path.join(pre_folder, f"cleaned_v{version}.parquet")
    cleaned_df.to_parquet(cleaned_path, index=False)
    date_columns = [
        col for col, dt in cleaned_df.dtypes.items()
        if str(dt).startswith("datetime64")
    ]

    metadata_path = os.path.join(pre_folder, f"metadata_v{version}.json")
    with open(metadata_path, "w") as f:
        json.dump({
            "numeric_continuous": num_cont,
            "numeric_categorical": num_cat,
            "categorical_columns": cat_cols,
            "date_columns": date_columns,
            "target_column": target_column
        }, f, indent=2)

    return {
        "status": "success",
        "version": version,
        "cleaned_path": cleaned_path,
        "metadata_path": metadata_path,
        "numeric_continuous": num_cont,
        "numeric_categorical": num_cat,
        "categorical_columns": cat_cols,
        "date_columns": date_columns,
    }


@app.function(image=ai_image, volumes={"/root/data": volume})
def feature_engineering_agent(
    user_id: str,
    target_column: str,
    fe_config: dict | None = None,
):
    """
    Run feature engineering for the latest cleaned dataset for this user.

    Parameters
    ----------
    user_id : str
        Identifier used to resolve the user's data folders on the shared volume.
    target_column : str
        Name of the target column in the dataframe.
    fe_config : dict | None
        Feature-engineering configuration controlling which steps to apply.
        If None, defaults from pipeline.Feature_Engineering.DEFAULT_FE_CONFIG are used.
    """
    import os
    import json
    import pandas as pd

    from pipeline.Feature_Engineering import run_feature_engineering
    from pipeline.utils_paths import get_user_preprocessed, get_user_engineered

    pre_folder = get_user_preprocessed(user_id)
    cleaned_files = [
        f for f in os.listdir(pre_folder)
        if f.startswith("cleaned_") and f.endswith(".parquet")
    ]
    if not cleaned_files:
        return {"status": "error", "message": "No cleaned files found."}

    latest_cleaned = max(
        cleaned_files,
        key=lambda f: int(f.split("_v")[1].split(".")[0])
    )
    version = int(latest_cleaned.split("_v")[1].split(".")[0])

    cleaned_path = os.path.join(pre_folder, latest_cleaned)
    df = pd.read_parquet(cleaned_path)

    engineered_df = run_feature_engineering(
        df=df,
        target_column=target_column,
        fe_config=fe_config,
    )

    eng_folder = get_user_engineered(user_id)
    os.makedirs(eng_folder, exist_ok=True)

    eng_path = os.path.join(eng_folder, f"engineered_v{version}.parquet")
    engineered_df.to_parquet(eng_path, index=False)

    if fe_config is not None:
        config_path = os.path.join(eng_folder, f"fe_config_v{version}.json")
        with open(config_path, "w") as f:
            json.dump(fe_config, f, indent=2)

    return {
        "status": "success",
        "version": version,
        "engineered_path": eng_path,          
        "shape": list(engineered_df.shape),
        "columns": engineered_df.columns.tolist(),
        "fe_config": fe_config,
    }

from pipeline.Feature_Engineering import handle_target_imbalance

def model_training_agent(
    user_id: str,
    target_column: str,
    model_list: list
):
    """
    model_list = [
        {"model_name": "lightgbm_regressor", "params": {...}, "experiment_name": "exp_name_optional"},
        ...
    ]
    """

    from pipeline.Model_utils import (
        agentic_train_test_split,
        train_linear_regression,
        train_random_forest_regressor,
        train_xgboost_regressor,
        train_lightgbm_regressor,
        train_gradient_boost_regressor,
        train_catboost_regressor,
        train_knn_regressor,
        train_logistic_regression,
        train_random_forest_classifier,
        train_xgboost_classifier,
        train_lightgbm_classifier,
        train_gradient_boost_classifier,
        train_catboost_classifier,
        train_knn_classifier
    )
    from pipeline.utils_paths import get_user_engineered, get_user_models
    import json
    import joblib
    import numpy as np
    import os
    import shutil
    import re
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder

    eng_folder = get_user_engineered(user_id)
    model_folder = get_user_models(user_id)
    os.makedirs(model_folder, exist_ok=True)

    engineered_files = [
        f for f in os.listdir(eng_folder)
        if f.startswith("engineered_v") and f.endswith(".parquet")
    ]
    if not engineered_files:
        return {"status": "error", "message": "No engineered dataset available."}

    latest_engineered = max(
        engineered_files,
        key=lambda f: int(f.split("_v")[1].split(".")[0])
    )
    version = int(latest_engineered.split("_v")[1].split(".")[0])

    engineered_path = os.path.join(eng_folder, latest_engineered)
    df = pd.read_parquet(engineered_path)


    def _sanitize_feature_name(name: str) -> str:
        """Replace special characters so LightGBM is happy."""
        name = name.replace(" ", "_")
        name = re.sub(r"[^0-9A-Za-z_]+", "_", name)
        name = re.sub(r"_+", "_", name).strip("_")
        return name[:80]
    feature_cols = [c for c in df.columns if c != target_column]

    # Build a collision-safe rename map
    rename_map = {}
    used_names: set[str] = set()

    for col in feature_cols:
        base = _sanitize_feature_name(col)
        new_name = base
        idx = 1

        # Ensure uniqueness and avoid clashing with target_column
        while new_name in used_names or new_name == target_column:
            idx += 1
            new_name = f"{base}__{idx}"

        rename_map[col] = new_name
        used_names.add(new_name)

    df = df.rename(columns=rename_map)


    # Determine task using both dtype and cardinality
    y_raw = df[target_column]
    n_unique = int(y_raw.nunique(dropna=True))
    is_numeric = np.issubdtype(y_raw.dtype, np.number)

    # Heuristic:
    # - numeric + few unique values => classification (e.g. 0/1 cardio)
    # - numeric + many unique values => regression (continuous target)
    # - non-numeric => classification (and we label-encode)
    if is_numeric and n_unique <= 10:
        task = "classification"
        # keep numeric 0/1 etc. as-is; no encoding needed
    elif is_numeric:
        task = "regression"
    else:
        task = "classification"
        # Encode string/categorical labels to integers
        le = LabelEncoder()
        df[target_column] = le.fit_transform(y_raw)

        # Save class mapping so predictions can be decoded later if needed
        label_map_path = os.path.join(
            model_folder,
            f"label_encoder_v{version}_{target_column}.json"
        )
        with open(label_map_path, "w") as f:
            json.dump(
                {
                    "target_column": target_column,
                    "classes": le.classes_.tolist(),
                },
                f,
                indent=2,
            )
        # ---- Class imbalance info (for class_weight / scale_pos_weight heuristics) ----
    class_counts = None
    class_ratios = None
    is_imbalanced = False
    binary_scale_pos_weight = None

    if task == "classification":
        # Use current df[target_column] (encoded if we used LabelEncoder)
        y_for_dist = df[target_column]
        class_counts = y_for_dist.value_counts()
        total = int(class_counts.sum())

        if total > 0 and len(class_counts) >= 2:
            class_ratios = (class_counts / total).round(4)
            minority_ratio = float(class_ratios.min())
            majority_ratio = float(class_ratios.max())
            is_imbalanced = minority_ratio < 0.2  # threshold tweakable

            # Only compute scale_pos_weight for numeric binary {0, 1} + imbalanced
            unique_vals = sorted(y_raw.dropna().unique().tolist())
            is_binary_01 = (
                is_numeric
                and len(unique_vals) == 2
                and set(unique_vals) <= {0, 1}
            )

            if is_binary_01 and is_imbalanced:
                n_pos = (y_raw == 1).sum()
                n_neg = (y_raw == 0).sum()
                if n_pos > 0:
                    binary_scale_pos_weight = float(n_neg / n_pos)
                else:
                    binary_scale_pos_weight = 1.0

    def _apply_class_weight_heuristics(
    model_list: list,
    is_classification: bool,
    is_imbalanced: bool,
    binary_scale_pos_weight: float | None,
    ) -> list:
        """
        If classification is imbalanced:
        - Set class_weight='balanced' for models that support it (if not provided).
        - For binary 0/1 and imbalanced, set scale_pos_weight for XGBoost / CatBoost
          if not already provided.
        """
        if not is_classification or not is_imbalanced:
            return model_list

        new_models: list[dict] = []

        for cfg in model_list:
            if not isinstance(cfg, dict):
                new_models.append(cfg)
                continue

            m = dict(cfg)
            params = dict(m.get("params") or {})
            name = m.get("model_name")

            # 1) class_weight='balanced' for supported classifiers
            if name in {
                "logistic_regression",
                "random_forest_classifier",
                "gradient_boost_classifier",
                "lightgbm_classifier",
            }:
                # Do not override if LLM/user explicitly set it
                params.setdefault("class_weight", "balanced")

            # 2) scale_pos_weight for binary 0/1 XGBoost / CatBoost
            if binary_scale_pos_weight is not None and name in {
                "xgboost_classifier",
                "catboost_classifier",
            }:
                params.setdefault("scale_pos_weight", binary_scale_pos_weight)

            m["params"] = params
            new_models.append(m)

        return new_models

    X_train, X_test, y_train, y_test = agentic_train_test_split(
        df,
        target_column,
        test_size=0.2,                # or expose this as a config later
        random_state=42,
        stratify=(task == "classification"),
        is_classification=(task == "classification"),
    )


    # Load fe_config for this version
    eng_folder = get_user_engineered(user_id)
    fe_config_path = os.path.join(eng_folder, f"fe_config_v{version}.json")
    fe_cfg = {}
    if os.path.exists(fe_config_path):
        with open(fe_config_path, "r") as f:
            fe_cfg = json.load(f)

    strategy = fe_cfg.get("imbalance_strategy", "none")

    if task == "classification" and strategy != "none":
        df_train = X_train.copy()
        df_train[target_column] = y_train.values

        df_train_balanced = handle_target_imbalance(
            df=df_train,
            target_column=target_column,
            strategy=strategy,
            sampling_strategy=fe_cfg.get("imbalance_sampling_strategy", "auto"),
            smote_k_neighbors=fe_cfg.get("imbalance_smote_k_neighbors", 5),
            random_state=fe_cfg.get("imbalance_random_state", 42),
            minority_threshold=fe_cfg.get("imbalance_minority_threshold", 0.2),
        )

        X_train = df_train_balanced.drop(columns=[target_column])
        y_train = df_train_balanced[target_column]


    REGRESSORS = {
        "linear_regression": train_linear_regression,
        "random_forest_regressor": train_random_forest_regressor,
        "xgboost_regressor": train_xgboost_regressor,
        "lightgbm_regressor": train_lightgbm_regressor,
        "gradient_boost_regressor": train_gradient_boost_regressor,
        "catboost_regressor": train_catboost_regressor,
        "knn_regressor": train_knn_regressor,
    }

    CLASSIFIERS = {
        "logistic_regression": train_logistic_regression,
        "random_forest_classifier": train_random_forest_classifier,
        "xgboost_classifier": train_xgboost_classifier,
        "lightgbm_classifier": train_lightgbm_classifier,
        "gradient_boost_classifier": train_gradient_boost_classifier,
        "catboost_classifier": train_catboost_classifier,
        "knn_classifier": train_knn_classifier,
    }

    results = []
    best_model_name = None
    best_model_path = None
    best_metrics_path = None
    best_generalization_gap = None
    
    # After optional resampling, adjust model params for class_weight / scale_pos_weight
    model_list = _apply_class_weight_heuristics(
        model_list=model_list,
        is_classification=(task == "classification"),
        is_imbalanced=is_imbalanced,
        binary_scale_pos_weight=binary_scale_pos_weight,
    )

    for model_cfg in model_list:
        model_name = model_cfg.get("model_name")
        params = model_cfg.get("params", {})
        experiment_name = model_cfg.get("experiment_name")

        if task == "regression":
            train_func = REGRESSORS.get(model_name)
        else:
            train_func = CLASSIFIERS.get(model_name)

        if train_func is None:
            continue

        result = train_func(
            X_train, y_train,
            X_test, y_test,
            params,
            experiment_name
        )

        model = result["model"]
        train_metrics = result["train_metrics"]
        test_metrics = result["test_metrics"]

        model_path = os.path.join(
            model_folder,
            f"model_v{version}_{model_name}.pkl"
        )
        joblib.dump(model, model_path)

        metrics_obj = {
            "train": train_metrics,
            "test": test_metrics,
            "params": result.get("params", params),
        }
        metrics_path = os.path.join(
            model_folder,
            f"metrics_v{version}_{model_name}.json"
        )
        with open(metrics_path, "w") as f:
            json.dump(metrics_obj, f, indent=2)

        if task == "regression":
            train_metric = train_metrics["rmse"]
            test_metric = test_metrics["rmse"]
            generalization_gap = test_metric - train_metric
            score_for_selection = test_metric  
        else:
            train_metric = train_metrics["accuracy"]
            test_metric = test_metrics["accuracy"]
            generalization_gap = train_metric - test_metric
            score_for_selection = -test_metric  

        results.append(
            {
                "model_name": model_name,
                "model_path": model_path,
                "metrics_path": metrics_path,
                "train_metrics": train_metrics,
                "test_metrics": test_metrics,
                "generalization_gap": generalization_gap,
                "score_for_selection": score_for_selection,
            }
        )


        # 1) Compute score_for_selection for this model
        if task == "regression":
            train_metric = train_metrics["rmse"]
            test_metric = test_metrics["rmse"]          # >0 = worse on test
            generalization_gap = test_metric - train_metric
            # Soft penalty: mostly care about test RMSE, add a small cost for overfitting
            gap_penalty = max(0.0, generalization_gap)
            score_for_selection = test_metric + 0.3 * gap_penalty   # lower is better

        else:  # classification
            train_metric = train_metrics["accuracy"]
            test_metric = test_metrics["accuracy"]
            generalization_gap = train_metric - test_metric         # >0 = overfitting
            gap_penalty = max(0.0, generalization_gap)
            # We want higher accuracy → lower score = better
            score_for_selection = -test_metric + 0.3 * gap_penalty

        # 2) Update best model using this score
        if best_model_name is None or score_for_selection < best_score:
            best_model_name = model_name
            best_model_path = model_path
            best_metrics_path = metrics_path
            best_generalization_gap = generalization_gap
            best_score = score_for_selection

    return {
        "status": "success",
        "version": version,
        "task": task,
        "results": results,
        "best_model_name": best_model_name,
        "best_model_path": best_model_path,
        "best_metrics_path": best_metrics_path,
        "generalization_gap": best_generalization_gap
    }




@app.function(image=ai_image, volumes={"/root/data": volume})
def prediction_agent(
    user_id: str,
    target_column: str,
    model_name: str = None
):

    from pipeline.utils_paths import get_user_engineered, get_user_models

    eng_folder = get_user_engineered(user_id)
    engineered_files = [
        f for f in os.listdir(eng_folder)
        if f.startswith("engineered_") and f.endswith(".parquet")
    ]
    if not engineered_files:
        return {"status": "error", "message": "No engineered dataset."}

    latest_eng = max(engineered_files, key=lambda f: int(f.split("_v")[1].split(".")[0]))
    version = int(latest_eng.split("_v")[1].split(".")[0])

    df = pd.read_parquet(os.path.join(eng_folder, latest_eng))
    X = df.drop(columns=[target_column])

    model_folder = get_user_models(user_id)

    if model_name is None:
        model_path = os.path.join(model_folder, f"model_v{version}_best.pkl")
    else:
        model_path = os.path.join(model_folder, f"model_v{version}_{model_name}.pkl")

    if not os.path.exists(model_path):
        return {"status": "error", "message": f"Model file not found: {model_path}"}

    import joblib
    model = joblib.load(model_path)
    preds = model.predict(X)

    preds_path = os.path.join(model_folder, f"predictions_v{version}.json")
    with open(preds_path, "w") as f:
        json.dump(preds.tolist(), f, indent=2)

    return {
        "status": "success",
        "version": version,
        "model_used": model_name or "best",
        "predictions_path": preds_path
    }


@app.function(image=ai_image, volumes={"/root/data": volume})
def visualization_agent(user_id: str):

    from pipeline.visualizations_utils import generate_dashboard_data
    from pipeline.utils_paths import (
        get_user_preprocessed,
        get_user_visualizations,
        get_user_uploads,
    )

    pre_folder = get_user_preprocessed(user_id)
    cleaned_files = [
        f for f in os.listdir(pre_folder)
        if f.startswith("cleaned_") and f.endswith(".parquet")
    ]
    if not cleaned_files:
        return {"status": "error", "message": "No cleaned files."}

    latest_cleaned = max(cleaned_files, key=lambda f: int(f.split("_v")[1].split(".")[0]))
    version = int(latest_cleaned.split("_v")[1].split(".")[0])

    cleaned_path = os.path.join(pre_folder, latest_cleaned)
    metadata_path = os.path.join(pre_folder, f"metadata_v{version}.json")

    if not os.path.exists(metadata_path):
        return {"status": "error", "message": "Metadata missing."}

    df = pd.read_parquet(cleaned_path)
    meta = json.load(open(metadata_path))
    uploads_folder = get_user_uploads(user_id)
    raw_path = os.path.join(uploads_folder, f"ingested_v{version}.parquet")
    df_raw = None
    if os.path.exists(raw_path):
        df_raw = pd.read_parquet(raw_path)

    viz_folder = get_user_visualizations(user_id)
    latest_dash = os.path.join(viz_folder, "dashboard_data.json")
    versioned_dash = os.path.join(viz_folder, f"dashboard_data_v{version}.json")

    generate_dashboard_data(
        df=df,
        user_id=user_id,
        numeric_continuous=meta["numeric_continuous"],
        numeric_categorical=meta["numeric_categorical"],
        categorical_columns=meta["categorical_columns"],
        target_column=meta.get("target_column"),
        date_columns=meta.get("date_columns", []),
        df_raw_for_counts=df_raw
    )

    shutil.copyfile(latest_dash, versioned_dash)

    return {
        "status": "success",
        "version": version,
        "latest_dashboard": latest_dash,
        "versioned_dashboard": versioned_dash
    }


@app.function(image=ai_image, volumes={"/root/data": volume})
def evaluation_agent(user_id: str) -> dict:

    import os
    import json
    from pipeline.utils_paths import get_user_models

    model_folder = get_user_models(user_id)

    if not os.path.exists(model_folder):
        return {
            "status": "error",
            "stage": "evaluation",
            "message": f"Model folder does not exist for user '{user_id}'.",
        }

    def _metrics_version_from_name(fname: str) -> int:
        try:
            after_v = fname.split("_v", 1)[1]      
            version_str = after_v.split("_", 1)[0] 
            return int(version_str)
        except Exception:
            return -1

    metrics_files = [
        f
        for f in os.listdir(model_folder)
        if f.startswith("metrics_v") and f.endswith(".json")
    ]

    metrics_files = [f for f in metrics_files if _metrics_version_from_name(f) >= 0]

    if not metrics_files:
        return {
            "status": "error",
            "stage": "evaluation",
            "message": "No metrics_v*.json files found for this user.",
        }

    metrics_files.sort(key=_metrics_version_from_name)

    history: dict[int, list[dict]] = {}

    for fname in metrics_files:
        version = _metrics_version_from_name(fname)
        if version < 0:
            continue

        if version not in history:
            history[version] = []

        path = os.path.join(model_folder, fname)
        try:
            with open(path, "r") as f:
                data = json.load(f)
        except Exception as e:
            data = {"error": f"Failed to load metrics: {e!r}"}

        history[version].append({
            "file": fname,
            "metrics": data,
        })

    return {
        "status": "success",
        "history": history,
    }

from pipeline.Preprocessing import clean_data
from pipeline.utils_paths import (
    get_user_uploads,
    get_user_preprocessed,
    get_user_visualizations,
    get_user_reports,
    get_user_dashboard_json,
)
from pipeline.visualizations_utils import (
    ALLOWED_CHART_TYPES,
    build_ready_charts_from_plan,
    json_safe,
)


from pipeline.Model_utils import build_model_list_from_config


@app.function(
    image=ai_image,
    volumes={"/root/data": volume},
    secrets=[groq_secret],
)
def auto_ai_agent(
    user_id: str,
    target_column: str,
    objective: str = "auto",
    run_automl: bool = False,
) -> dict:
    """
    High-level GenAI agent with two phases:

    Phase 1 (run_automl=False):
      - Use LLM to decide how to CLEAN the data (preprocess_config) from RAW,
        unless cleaned_vN.parquet already exists (reuse if so; no double cleaning).
      - Clean & save cleaned_vN.parquet + metadata_vN.json.
      - Run visualization_agent on CLEANED data to build EDA JSON.
      - Use LLM to decide a visualization plan (viz_plan) for Dash layout on CLEANED data.
      - Save viz_plan_vN.json and return configs + version + dashboard URL hint.

    Phase 2 (run_automl=True):
      - Reuse existing CLEANED data (no re-cleaning).
      - Use LLM to decide feature engineering (fe_config) + models (model_list).
      - Run feature_engineering_agent + model_training_agent.
      - Return training results + configs + version + dashboard URL hint.
    """
    uploads = get_user_uploads(user_id)
    version = _detect_latest_version(uploads, "ingested_", ".parquet")
    if version is None:
        return {
            "status": "error",
            "stage": "ingestion",
            "message": "No ingested_vN.parquet found for this user.",
        }

    raw_path = os.path.join(uploads, f"ingested_v{version}.parquet")
    if not os.path.exists(raw_path):
        return {
            "status": "error",
            "stage": "ingestion",
            "message": f"Missing {raw_path}",
        }

    df_raw = pd.read_parquet(raw_path)
    if target_column not in df_raw.columns:
        return {
            "status": "error",
            "stage": "ingestion",
            "message": f"Target column '{target_column}' not in dataset.",
        }

    n_rows, n_cols = df_raw.shape
    dtypes = {col: str(dt) for col, dt in df_raw.dtypes.items()}
    missing_ratios = df_raw.isnull().mean().round(4).to_dict()
    n_uniques = df_raw.nunique().to_dict()

    target_series = df_raw[target_column]
    target_dtype = str(target_series.dtype)
    n_unique = int(target_series.nunique())

    is_numeric = target_dtype.startswith(("float", "int"))

    if is_numeric and n_unique <= 10:
        task = "classification"
    else:
        task = "regression" if is_numeric else "classification"


    raw_summary = {
        "user_id": user_id,
        "version": version,
        "rows": n_rows,
        "cols": n_cols,
        "target_column": target_column,
        "target_dtype": target_dtype,
        "task": task,
        "objective": objective,
        "dtypes": dtypes,
        "missing_ratios": missing_ratios,
        "n_unique": n_uniques,
    }


    pre_folder = get_user_preprocessed(user_id)
    os.makedirs(pre_folder, exist_ok=True)

    cleaned_path = os.path.join(pre_folder, f"cleaned_v{version}.parquet")
    metadata_path = os.path.join(pre_folder, f"metadata_v{version}.json")

    def _default_preprocess_config(task_: str) -> dict:
        return {
            "set_id_index": True,
            "remove_duplicates": True,
            "handle_dates": True,
            "handle_missing": True,
            "missing_col_threshold": 0.5,
            "missing_row_threshold": 0.05,
            "outlier_strategy": "iqr",
            "iqr_multiplier": 1.5,
        }


    preprocess_config: dict | None = None
    preprocess_llm_error: str | None = None
    decision_report: str | None = None
    report_path: str | None = None


    if os.path.exists(cleaned_path) and os.path.exists(metadata_path):
        cleaned_df = pd.read_parquet(cleaned_path)
        with open(metadata_path, "r") as f:
            meta = json.load(f)
        num_cont = meta.get("numeric_continuous", [])
        num_cat = meta.get("numeric_categorical", [])
        cat_cols = meta.get("categorical_columns", [])
        preprocess_config = meta.get("preprocess_config") or _default_preprocess_config(task)
    else:
        preprocess_config = _default_preprocess_config(task)

        try:
            client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

            system_msg = (
                """You are a SENIOR DATA ENGINEER specializing in data preprocessing for ML pipelines.

            TASK: Analyze the raw dataset and propose optimal preprocessing configuration.

            DECISION GUIDE:
            ═══════════════════════════════════════════════════════════════════════════════

            SET_ID_INDEX:
            - True: If obvious ID column exists (id, index, row_id, etc.)
            - False: If no clear ID column

            REMOVE_DUPLICATES:
            - True: Almost always (default)
            - False: Only if duplicates have meaning (time-series events)

            HANDLE_DATES:
            - True: If datetime columns exist
            - False: If no date columns

            MISSING_COL_THRESHOLD (0.0-1.0):
            - Small datasets (<1000): Use 0.7-0.8 (preserve more data)
            - Large datasets (>10000): Use 0.4-0.5 (stricter)

            MISSING_ROW_THRESHOLD (0.0-1.0):
            - Typically 0.3-0.5 works well
            - Be conservative with small datasets

            OUTLIER_STRATEGY:
            - "iqr": Standard method (good for most cases)
            - "none": Small datasets or meaningful outliers

            IQR_MULTIPLIER:
            - 1.5: Aggressive (removes more)
            - 2.0: Moderate
            - 3.0: Conservative (removes less)
            - Small datasets: Use 2.0-3.0
            - Large datasets: Use 1.5-2.0


            OUTPUT FORMAT:
            Return ONLY JSON with key 'preprocess_config':
            {
            "preprocess_config": {
                "set_id_index": bool,
                "remove_duplicates": bool,
                "handle_dates": bool,
                "handle_missing": bool,
                "missing_col_threshold": float,
                "missing_row_threshold": float,
                "outlier_strategy": "iqr" | "none",
                "iqr_multiplier": float,
            }
            }
            """
            )

            user_msg = (
                f"""
                DATASET ANALYSIS:
                - Rows: {n_rows}
                - Columns: {n_cols}
                - Task: {task}
                - Target: {target_column} ({target_dtype})
                - High missing columns: {[k for k,v in missing_ratios.items() if v > 0.3]}

                FULL SUMMARY:
                {json.dumps(raw_summary, indent=2)}

                Based on this analysis, provide the optimal preprocess_config.
                """
            )

            completion = client.chat.completions.create(
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.2,
                response_format={"type": "json_object"},
            )
            content = completion.choices[0].message.content
            obj = json.loads(content)

            if isinstance(obj, dict) and "preprocess_config" in obj and isinstance(
                obj["preprocess_config"], dict
            ):
                preprocess_config = obj["preprocess_config"]
        except Exception as e:
            preprocess_llm_error = str(e)

        cfg = preprocess_config or {}
        cfg.setdefault("set_id_index", True)
        cfg.setdefault("remove_duplicates", True)
        cfg.setdefault("handle_dates", True)
        cfg.setdefault("handle_missing", True)
        cfg.setdefault("missing_col_threshold", 0.5)
        cfg.setdefault("missing_row_threshold", 0.05)
        cfg.setdefault("outlier_strategy", "iqr")
        cfg.setdefault("iqr_multiplier", 1.5)


        try:
            cfg["missing_col_threshold"] = float(cfg["missing_col_threshold"])
        except Exception:
            cfg["missing_col_threshold"] = 0.5
        cfg["missing_col_threshold"] = max(0.0, min(1.0, cfg["missing_col_threshold"]))

        try:
            cfg["missing_row_threshold"] = float(cfg["missing_row_threshold"])
        except Exception:
            cfg["missing_row_threshold"] = 0.05
        cfg["missing_row_threshold"] = max(0.0, min(1.0, cfg["missing_row_threshold"]))

        if cfg.get("outlier_strategy") not in ("iqr", "none"):
            cfg["outlier_strategy"] = "iqr"

        try:
            cfg["iqr_multiplier"] = float(cfg["iqr_multiplier"])
        except Exception:
            cfg["iqr_multiplier"] = 1.5

        preprocess_config = cfg

        cleaned_df, num_cont, num_cat, cat_cols = clean_data(
            df_raw,
            target_column=target_column,
            config=preprocess_config,
        )
        date_columns = [
            col for col, dt in cleaned_df.dtypes.items()
            if str(dt).startswith("datetime64")
        ]
        cleaned_df.to_parquet(cleaned_path, index=False)
        with open(metadata_path, "w") as f:
            json.dump(
                {
                    "numeric_continuous": num_cont,
                    "numeric_categorical": num_cat,
                    "categorical_columns": cat_cols,
                    "date_columns": date_columns,
                    "target_column": target_column,
                    "preprocess_config": preprocess_config,
                },
                f,
                indent=2,
            )

    viz_result = visualization_agent.local(user_id)
    if viz_result.get("status") != "success":
        return {
            "status": "error",
            "stage": "visualization",
            "detail": viz_result,
            "version": version,
            "preprocess_config": preprocess_config,
            "preprocess_llm_error": preprocess_llm_error,
        }

    dashboard_path = viz_result["latest_dashboard"]
    with open(dashboard_path, "r") as f:
        dashboard_data = json.load(f)


    viz_llm_error: str | None = None

    viz_context = {
        "version": version,
        "rows": int(cleaned_df.shape[0]),
        "cols": int(cleaned_df.shape[1]),
        "columns": list(cleaned_df.columns),
        "target_column": target_column,
        "task": task,
        "objective": objective,
        "missing_summary": dashboard_data.get("missing_values", {}),
    }

    DEFAULT_VIZ_PLAN = {
        "charts": [
            {
                "id": "target_hist",
                "type": "histogram",
                "x": target_column,
                "y": None,
                "color": None,
                "description": f"Distribution of {target_column}",
                "width": 12,
                "height": 1,
                "order": 1,
            }
        ]
    }

    viz_plan = DEFAULT_VIZ_PLAN

    try:
        client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

        system_msg = (
    """
    You are a visualization-planning assistant for a data science dashboard.
    You must output a JSON object with a list of charts following STRICT rules.

    You will be given a DATASET SUMMARY (viz_context) as JSON. It may contain:
    - "columns": list of all column names
    - "target_column": name of the target column (or null)
    - "task": "regression" or "classification"
    - "numeric_continuous": list of continuous numeric features
    - "numeric_categorical": list of numeric but low-cardinality features (treat as categorical)
    - "categorical_columns": list of categorical features
    - other stats (missing values, etc.)

    INTERPRETATION RULES (must match downstream code):
    - Columns in "numeric_continuous":
        * Use as numeric for histogram, scatter, box (as y), line, and correlation heatmap.
    - Columns in "numeric_categorical" or "categorical_columns":
        * Treat as categorical (use for bar / pie x-axis, or as x in box plots).
    - Target type:
        * If target_column is in numeric_continuous → treat target as a CONTINUOUS numeric target.
        * Otherwise (target in numeric_categorical / categorical_columns OR task == "classification")
          → treat target as CATEGORICAL. Even if it’s stored as int (e.g. 0/1), think of it as a label.

    STRICT RULES:
    1. Every chart MUST have:
       - id (string)
       - type (one of: "histogram", "scatter", "bar", "box", "heatmap", "pie", "line")
       - description (string)
       - width (3, 4, 6, or 12)
       - order (int)
       Optional keys: x, y, color (depending on chart type rules below).

    2. DO NOT EVER produce null x or y unless allowed by rules below.
       - Allowed to omit:
         * histogram: include x only, no y key.
         * bar / pie: may omit y to mean "count of x".
         * heatmap: no x and no y keys at all.

    3. Type rules (must respect column types from viz_context):
       - histogram:
           * requires x only
           * DO NOT include y
           * x must be a numeric column from numeric_continuous
       - scatter:
           * requires x and y
           * BOTH must be numeric_continuous columns
           * NEVER use a categorical or numeric_categorical column on x or y
       - bar:
           * requires x (categorical)
           * x must come from categorical_columns, numeric_categorical, or a CATEGORICAL target
           * y is optional; if present it must be numeric_continuous
       - box:
           * requires y (numeric_continuous)
           * x optional; if present it must be categorical (categorical_columns, numeric_categorical, or categorical target)
       - pie:
           * requires x (categorical)
           * x must be categorical (categorical_columns, numeric_categorical, or categorical target)
           * DO NOT include y
       - heatmap:
           * no x/y required
           * always reflect correlations among numeric_continuous columns
       - line:
           * requires x (date/datetime or a sequential numeric column)
           * requires y (numeric_continuous)
           * used for time-series or trend visualization

    4. Target-aware rules (decide behaviour from target type):

       4.a CONTINUOUS numeric target (regression-like, target in numeric_continuous):
           - MUST include at least:
             * 1 histogram of the target (type="histogram", x=target_column).
             * 3+ scatter plots: numeric_continuous feature vs target_column (x=feature, y=target_column).
             * 2+ bar charts: x = each categorical / numeric_categorical feature,
               y = target_column (mean target by category).
             * If there is a date / time or clearly sequential column in columns,
               include at least 1 line chart with x=time-like, y=target_column.

       4.b CATEGORICAL target (classification-like, target NOT in numeric_continuous):
           - NEVER treat target as numeric. Do NOT:
             * use target as y in scatter or histogram,
             * or use target as y in any chart requiring numeric y.
           - MUST include at least:
             * 1 bar chart of target distribution:
               - type="bar", x=target_column, no y (counts).
             * 2+ box plots of numeric_continuous features by target:
               - type="box", x=target_column, y=<numeric_continuous>.
             * 2+ bar charts combining target with other categorical features:
               - Example: x=<other categorical or numeric_categorical>, color=target_column, no y (counts).
           - Optionally, you may use target_column as:
             * x for bar / pie charts, or
             * color grouping for scatter/box/line (but only when y is numeric_continuous).

5. Mix of target-focused and feature–feature charts:
       - Create **exactly 6 charts** in total.
       - Do NOT create more than 6 charts.
       - At least **50%** of charts MUST involve target_column in x, y, or color.
       - Include at least **3 charts** that do NOT use target_column at all:
         * e.g. correlation heatmap over numeric_continuous,
         * histograms of important numeric_continuous features,
         * bar/pie charts of key categorical_columns.
       - Ensure at least **one** chart is a correlation heatmap (type="heatmap")
         over numeric_continuous columns.

    6. Column validity:
       - Only use columns that ACTUALLY exist in the dataset (see the DATASET SUMMARY).
       - For numeric axes (histogram, scatter, box y, line y), pick from numeric_continuous only.
       - For categorical axes (bar x, pie x, box x), pick from categorical_columns,
         numeric_categorical, or a CATEGORICAL target.

    7. Output format:
       - Return ONLY valid JSON with the top-level key "charts", e.g.:
         {
           "charts": [
             {
               "id": "...",
               "type": "...",
               "description": "...",
               "width": 6,
               "order": 1,
               "x": "...",
               "y": "...",
               "color": "..."
             },
             ...
           ]
         }
    """
)



        user_msg = (
            f"""
            DATASET SUMMARY:
            {json.dumps(viz_context, indent=2)}

            Create a concise professional dashboard with **exactly 6** visualizations
            that follow all the rules above.
            You MUST include:
            - at least one correlation heatmap (type="heatmap") over numeric_continuous columns
            - a mix of target-focused and feature–feature charts as specified.
            """
        )

        completion = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.3,
            response_format={"type": "json_object"},
        )
        content = completion.choices[0].message.content
        obj = json.loads(content)

        if "charts" in obj and isinstance(obj["charts"], list) and obj["charts"]:
            viz_plan = {"charts": obj["charts"]}
    except Exception as e:
        viz_llm_error = str(e)

    raw_charts = viz_plan.get("charts", []) or []
    validated_charts: list[dict] = []

    for chart in raw_charts:
        if not isinstance(chart, dict):
            continue

        ctype = chart.get("type")
        if ctype not in ALLOWED_CHART_TYPES:
            continue

        # Drop charts that refer to non-existent columns (except heatmap, which has no x/y)
        bad_col = False
        for col in [chart.get("x"), chart.get("y"), chart.get("color")]:
            if col is not None and col not in cleaned_df.columns:
                bad_col = True
                break
        if bad_col:
            continue

        # Normalize width
        w = chart.get("width", 6)
        allowed_w = [3, 4, 6, 12]
        if w not in allowed_w:
            w = min(allowed_w, key=lambda a: abs(a - (w or 6)))
        chart["width"] = w
        chart.setdefault("height", 1)
        chart.setdefault("order", 999)

        validated_charts.append(chart)

    # Ensure at least one correlation heatmap
    has_heatmap = any(c.get("type") == "heatmap" for c in validated_charts)
    if not has_heatmap:
        validated_charts.insert(
            0,
            {
                "id": "corr_heatmap",
                "type": "heatmap",
                "description": "Correlation heatmap of numeric features",
                "width": 12,
                "height": 1,
                "order": 1,
            },
        )

    # If LLM gave fewer than 6 valid charts, auto-fill with simple histograms
    # of numeric_continuous features (excluding target).
    existing_ids = {c.get("id") for c in validated_charts if c.get("id") is not None}
    used_x = {c.get("x") for c in validated_charts if c.get("x") is not None}

    for col in num_cont:
        if len(validated_charts) >= 6:
            break
        if col == target_column:
            continue
        if col in used_x:
            continue

        auto_id = f"auto_hist_{col}"
        if auto_id in existing_ids:
            continue

        validated_charts.append(
            {
                "id": auto_id,
                "type": "histogram",
                "description": f"Distribution of {col}",
                "x": col,
                "width": 6,
                "height": 1,
                "order": 999,
            }
        )
        existing_ids.add(auto_id)
        used_x.add(col)

    # Hard cap: exactly 6 charts if possible
    if validated_charts:
        validated_charts = validated_charts[:6]
        for idx, c in enumerate(validated_charts, start=1):
            c["order"] = idx
    else:
        # Fallback if everything failed – you may also expand DEFAULT_VIZ_PLAN to 6 charts if you want
        validated_charts = DEFAULT_VIZ_PLAN["charts"]

    viz_plan = {"charts": validated_charts}

    viz_folder = get_user_visualizations(user_id)
    os.makedirs(viz_folder, exist_ok=True)
    viz_plan_path = os.path.join(viz_folder, f"viz_plan_v{version}.json")
    latest_dash = os.path.join(viz_folder, "dashboard_data.json")
    versioned_dash = os.path.join(viz_folder, f"dashboard_data_v{version}.json")

    # Save raw LLM viz plan (for debugging / inspection)
    with open(viz_plan_path, "w") as f:
        json.dump(viz_plan, f, indent=2)

    try:
        ready_charts = build_ready_charts_from_plan(
            cleaned_df,
            viz_plan,
            num_cont,
            num_cat,
            cat_cols,
            target_column=target_column,
        )

        # Inject into dashboard_data (which we loaded earlier)
        dashboard_data["charts"] = ready_charts

        # 🔐 Make everything JSON-safe before dumping (handles Timestamp, numpy types, etc.)
        safe_dashboard_data = json_safe(dashboard_data)

        # Overwrite latest dashboard JSON and update the versioned snapshot
        with open(latest_dash, "w") as f:
            json.dump(safe_dashboard_data, f, indent=2)

        try:
            shutil.copyfile(latest_dash, versioned_dash)
        except Exception as copy_err:
            logger.warning(f"Failed to copy dashboard_data to versioned file: {copy_err!r}")

    except Exception as e:
        logger.warning(f"Failed to build ready-to-plot charts from LLM viz_plan: {e!r}")


    # ------------------------------------------------------------------
    # 4. If run_automl == False → Phase 1 only (clean + viz), stop here
    # ------------------------------------------------------------------
    if not run_automl:
        return {
            "status": "success",
            "phase": "data_only",
            "version": version,
            "task": task,
            "preprocess_config": preprocess_config,
            "preprocess_llm_error": preprocess_llm_error,
            "viz_llm_error": viz_llm_error,
            "viz_plan_path": viz_plan_path,
            "dashboard_url_hint": f"/dashboard?user_id={user_id}&version={version}",
        }


    def _default_fe_config(task_: str, has_cats: bool) -> dict:
        return {
            "encoding_strategy": (
                "target"
                if has_cats and task_ == "regression"
                else "one_hot"
                if has_cats
                else "none"
            ),
            "use_polynomial": False,
            "poly_degree": 2,
            "use_interactions": False,
            "use_binning": False,
            "use_pca": False,
            "imbalance_strategy": "none",
            "imbalance_sampling_strategy": "auto",
            "imbalance_smote_k_neighbors": 5,
            "imbalance_random_state": 42,
            "imbalance_minority_threshold": 0.2,
            }

    def _default_model_list(task_: str) -> list:
        if task_ == "regression":
            return [
                {
                    "model_name": "lightgbm_regressor",
                    "params": {},
                    "experiment_name": "auto_lgbm_reg",
                },
                {
                    "model_name": "random_forest_regressor",
                    "params": {},
                    "experiment_name": "auto_rf_reg",
                },
            ]
        else:
            return [
                {
                    "model_name": "lightgbm_classifier",
                    "params": {},
                    "experiment_name": "auto_lgbm_clf",
                },
                {
                    "model_name": "random_forest_classifier",
                    "params": {},
                    "experiment_name": "auto_rf_clf",
                },
            ]

    has_cats = bool(cat_cols)
    fe_config = _default_fe_config(task, has_cats)

    raw_model_list = _default_model_list(task)
    model_list = raw_model_list
    fe_llm_error: str | None = None
    target_distribution = None
    is_imbalanced = None

    if target_column in cleaned_df.columns and task == "classification":
        vc = cleaned_df[target_column].value_counts(dropna=False)
        total = int(vc.sum())
        ratios = (vc / total).round(4)

        minority_ratio = float(ratios.min())
        majority_ratio = float(ratios.max())

        target_distribution = {
            "counts": vc.to_dict(),
            "ratios": ratios.to_dict(),
            "total": total,
        }
        is_imbalanced = bool(minority_ratio < 0.2)


    cleaned_summary = {
        "rows": int(cleaned_df.shape[0]),
        "cols": int(cleaned_df.shape[1]),
        "numeric_continuous": num_cont,
        "numeric_categorical": num_cat,
        "categorical_columns": cat_cols,
        "target_column": target_column,
        "task": task,
        "objective": objective,
        "missing_overview": dashboard_data.get("missing_values", {}),
        "target_distribution": target_distribution,
        "is_imbalanced": is_imbalanced,
    }

    try:
        client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

        system_msg = (
            """You are a SENIOR MACHINE LEARNING ENGINEER with 10+ years of experience.
                You design feature engineering strategies and select ML models + hyperparameters
                for tabular datasets in a production AutoML system.

                You must obey the schema and rules below EXACTLY.
                Your output will be parsed by another program and INVALID JSON or EXTRA FIELDS
                will BREAK the pipeline.

                ===============================================================================
                INPUT YOU RECEIVE
                ===============================================================================
                You will receive a single USER message containing a JSON-like description
                of the dataset, similar to:

                {
                "rows": <int>,
                "cols": <int>,
                "numeric_continuous": [ ... ],
                "numeric_categorical": [ ... ],
                "categorical_columns": [ ... ],
                "target_column": "<name>",
                "task": "regression" | "classification",
                "objective": "auto" | "maximize_accuracy" | "minimize_rmse" | ...,
                "missing_overview": {...}
                "target_distribution": {
                    "counts": { "<class>": int, ... },
                    "ratios": { "<class>": float, ... },
                    "total": int
                },
                "is_imbalanced": true | false | null
                }
                }
                -If task == "classification" and is_imbalanced == true, you MUST consider imbalance handling.

                -For regression or balanced classification, you should usually leave imbalance handling as "none"
                
                - numeric_continuous: true numeric features (continuous)
                - numeric_categorical: numeric features that behave like categories
                - categorical_columns: object / category / bool / datetime
                - task: "regression" if target is numeric, "classification" otherwise


                ===============================================================================
                YOUR JOB (HIGH LEVEL)
                ===============================================================================
                1. Propose a FEATURE ENGINEERING CONFIG (fe_config) with:
                - encoding strategy for categorical variables
                - optional polynomial features
                - optional numeric interactions
                - optional binning
                - optional PCA

                2. Propose a MODEL LIST (model_list) with 3–4 DIVERSE models suitable for
                the task, each with concrete hyperparameters.

                3. You MUST follow the EXACT JSON SCHEMA described below.
                NO extra keys. NO comments. NO reasoning field. JSON ONLY.


                ===============================================================================
                OUTPUT SCHEMA (STRICT – DO NOT VIOLATE)
                ===============================================================================
                You MUST output a SINGLE JSON object with exactly these two top-level keys:

                {
                "fe_config": {
                    "encoding_strategy": "none" | "one_hot" | "label" | "ordinal" | "frequency" | "target",
                    "use_polynomial": true | false,
                    "poly_degree": 2 or 3,
                    "use_interactions": true | false,
                    "use_binning": true | false,
                    "bins": integer (e.g. 5 or 10),
                    "bin_strategy": "quantile" | "uniform",
                    "use_pca": true | false,
                    "pca_n_components": integer (e.g. 5, 10, 20),
                    "scale_numeric": true | false,
                    "imbalance_strategy": "none"
                        | "random_over"
                        | "random_under"
                        | "smote"
                        | "smotenc"
                        | "adasyn"
                        | "smoteenn"
                        | "smotetomek",

                    "imbalance_sampling_strategy": "auto" | float in (0, 1] | <class→ratio mapping>,
                    "imbalance_smote_k_neighbors": integer (e.g. 3–10),
                    "imbalance_random_state": integer,
                    "imbalance_minority_threshold": float in [0.05, 0.3]
                },
                },
                - For REGRESSION or when is_imbalanced is false or null:
                    * Set "imbalance_strategy": "none".
                - For CLASSIFICATION with is_imbalanced == true:
                    * Prefer "smotenc" when there are categorical columns.
                    * Prefer "smote" or "adasyn" when features are mostly numeric.
                    * Consider "smoteenn" or "smotetomek" when dataset is noisy and you want
                    both oversampling and cleaning.
                    * Use "random_over" for very small datasets or when you want a simple baseline.
                - Never oversample the test set: fe_config assumes resampling is applied on the
                training data only.


                "model_list": [
                    {
                    "model_name": "<one_of_allowed_names_below>",
                    "enabled": true,
                    "params": { <ONLY allowed hyperparameters for that model> },
                    "experiment_name": "<short_snake_case_name>"
                    },
                    ...
                ]
                }

                RULES:
                - Do NOT include any other top-level keys.
                - Do NOT include "reasoning", "notes", "description", "comment", etc.
                - Every model in model_list MUST include ALL 4 fields:
                model_name, enabled, params, experiment_name.
                - enabled MUST always be true for all returned models (this system only uses enabled models).
                - params MUST NOT be empty. You must set explicit values (e.g., n_estimators: 300).


                ===============================================================================
                MODEL NAME OPTIONS (YOU MUST USE THESE EXACT STRINGS)
                ===============================================================================
                For REGRESSION (task == "regression"), you may choose from:

                - "linear_regression"
                - "random_forest_regressor"
                - "xgboost_regressor"
                - "lightgbm_regressor"
                - "gradient_boost_regressor"
                - "catboost_regressor"
                - "knn_regressor"

                For CLASSIFICATION (task == "classification"), you may choose from:

                - "logistic_regression"
                - "random_forest_classifier"
                - "xgboost_classifier"
                - "lightgbm_classifier"
                - "gradient_boost_classifier"
                - "catboost_classifier"
                - "knn_classifier"

                Choose 3–5 models per call, including:
                - At least one tree ensemble (Random Forest or Gradient Boosting),
                - At least one boosting model (LightGBM or XGBoost),
                - At least one CatBoost model whenever there is ANY numeric_categorical or categorical feature:
                    * use "catboost_classifier" for classification
                    * use "catboost_regressor" for regression
                - Optionally Logistic Regression or KNN as an extra baseline.

                You MUST include a CatBoost model in model_list whenever:
                - task is "regression" or "classification", AND
                - numeric_categorical or categorical_columns is non-empty.
                ===============================================================================
                ALLOWED PARAMS PER MODEL (STRICT WHITELIST)
                ===============================================================================
                You MUST only use hyperparameters listed below for each model.
                If a parameter is not listed for that model, DO NOT include it.

                1) REGRESSION MODELS
                --------------------

                (1.a) "linear_regression"
                allowed params:
                    - fit_intercept (bool)
                    - copy_X (bool)
                    - n_jobs (int or null)
                    - positive (bool)

                (1.b) "random_forest_regressor"
                allowed params:
                    - n_estimators (int)
                    - criterion ("squared_error" | "absolute_error" | "friedman_mse" | "poisson")
                    - max_depth (int or null)
                    - min_samples_split (int)
                    - min_samples_leaf (int)
                    - min_weight_fraction_leaf (float)
                    - max_features ("sqrt" | "log2" | float in (0,1] | null)
                    - max_leaf_nodes (int or null)
                    - min_impurity_decrease (float)
                    - bootstrap (bool)
                    - oob_score (bool)
                    - n_jobs (int, often -1)
                    - random_state (int or null)
                    - max_samples (int or float or null)

                (1.c) "xgboost_regressor"
                allowed params:
                    - n_estimators (int)
                    - max_depth (int)
                    - learning_rate (float)
                    - subsample (float)
                    - colsample_bytree (float)
                    - gamma (float)
                    - min_child_weight (float)
                    - reg_alpha (float)
                    - reg_lambda (float)
                    - objective (string, e.g. "reg:squarederror")
                    - tree_method (string, e.g. "auto")
                    - n_jobs (int)
                    - early_stopping_rounds (int)
                    - eval_metric (string, e.g. "rmse")

                (1.d) "lightgbm_regressor"
                allowed params:
                    - n_estimators (int)
                    - num_leaves (int)
                    - learning_rate (float)
                    - subsample (float)
                    - colsample_bytree (float)
                    - min_child_samples (int)
                    - reg_alpha (float)
                    - reg_lambda (float)
                    - max_depth (int)
                    - min_split_gain (float)
                    - bagging_freq (int)
                    - objective (string, e.g. "regression")
                    - n_jobs (int)
                    - metric (string, e.g. "rmse")
                    - early_stopping_rounds (int)

                (1.e) "gradient_boost_regressor"
                allowed params:
                    - loss (string, e.g. "squared_error")
                    - learning_rate (float)
                    - n_estimators (int)
                    - subsample (float)
                    - criterion (string, e.g. "friedman_mse")
                    - min_samples_split (int)
                    - min_samples_leaf (int)
                    - min_weight_fraction_leaf (float)
                    - max_depth (int)
                    - min_impurity_decrease (float)
                    - max_features (int | float | "sqrt" | "log2" | null)
                    - alpha (float)
                    - n_iter_no_change (int or null)
                    - validation_fraction (float)

                (1.f) "catboost_regressor"
                allowed params:
                    - iterations (int)
                    - depth (int)
                    - learning_rate (float)
                    - l2_leaf_reg (float)
                    - rsm (float)
                    - loss_function (string, e.g. "RMSE")
                    - border_count (int)
                    - bagging_temperature (float)
                    - random_strength (float)
                    - grow_policy (string, e.g. "SymmetricTree")
                    - eval_metric (string, e.g. "RMSE")
                    - od_type (string, e.g. "Iter")
                    - od_wait (int)
                    - early_stopping_rounds (int)

                (1.g) "knn_regressor"
                allowed params:
                    - n_neighbors (int)
                    - weights ("uniform" | "distance")
                    - algorithm ("auto" | "ball_tree" | "kd_tree" | "brute")
                    - leaf_size (int)
                    - p (int, e.g. 1 or 2)
                    - metric ("minkowski" | "euclidean" | "manhattan")
                    - n_jobs (int)


                2) CLASSIFICATION MODELS
                ------------------------

                (2.a) "logistic_regression"
                allowed params:
                    - C (float)
                    - penalty (string, e.g. "l2")
                    - solver (string, e.g. "lbfgs", "saga")
                    - max_iter (int)
                    - fit_intercept (bool)
                    - class_weight (null or "balanced")
                    - n_jobs (int)
                    - l1_ratio (float or null)
                    - multi_class ("auto" | "ovr" | "multinomial")

                (2.b) "random_forest_classifier"
                allowed params:
                    - n_estimators (int)
                    - criterion ("gini" | "entropy" | "log_loss")
                    - max_depth (int or null)
                    - min_samples_split (int)
                    - min_samples_leaf (int)
                    - min_weight_fraction_leaf (float)
                    - max_features ("sqrt" | "log2" | float in (0,1] | null)
                    - max_leaf_nodes (int or null)
                    - min_impurity_decrease (float)
                    - bootstrap (bool)
                    - oob_score (bool)
                    - n_jobs (int)
                    - random_state (int or null)
                    - max_samples (int or float or null)
                    - class_weight (null | "balanced" | "balanced_subsample")

                (2.c) "xgboost_classifier"
                allowed params:
                    - n_estimators (int)
                    - max_depth (int)
                    - learning_rate (float)
                    - subsample (float)
                    - colsample_bytree (float)
                    - gamma (float)
                    - min_child_weight (float)
                    - reg_alpha (float)
                    - reg_lambda (float)
                    - objective (string, e.g. "binary:logistic" or "multi:softprob")
                    - tree_method (string, e.g. "auto")
                    - n_jobs (int)
                    - scale_pos_weight (float)
                    - eval_metric (string, e.g. "logloss")
                    - early_stopping_rounds (int)

                (2.d) "lightgbm_classifier"
                allowed params:
                    - n_estimators (int)
                    - num_leaves (int)
                    - learning_rate (float)
                    - subsample (float)
                    - colsample_bytree (float)
                    - min_child_samples (int)
                    - reg_alpha (float)
                    - reg_lambda (float)
                    - max_depth (int)
                    - min_split_gain (float)
                    - bagging_freq (int)
                    - objective (string, e.g. "binary")
                    - n_jobs (int)
                    - metric (string, e.g. "logloss")
                    - class_weight (null | "balanced")
                    - early_stopping_rounds (int)

                (2.e) "gradient_boost_classifier"
                allowed params:
                    - loss (string, e.g. "log_loss")
                    - learning_rate (float)
                    - n_estimators (int)
                    - subsample (float)
                    - criterion (string)
                    - min_samples_split (int)
                    - min_samples_leaf (int)
                    - min_weight_fraction_leaf (float)
                    - max_depth (int)
                    - min_impurity_decrease (float)
                    - max_features (int | float | "sqrt" | "log2" | null)
                    - n_iter_no_change (int or null)
                    - validation_fraction (float)

                (2.f) "catboost_classifier"
                allowed params:
                    - iterations (int)
                    - depth (int)
                    - learning_rate (float)
                    - l2_leaf_reg (float)
                    - rsm (float)
                    - loss_function (string, e.g. "Logloss")
                    - border_count (int)
                    - bagging_temperature (float)
                    - random_strength (float)
                    - grow_policy (string)
                    - eval_metric (string, e.g. "Logloss")
                    - scale_pos_weight (float)
                    - od_type (string, e.g. "Iter")
                    - od_wait (int)
                    - early_stopping_rounds (int)

                (2.g) "knn_classifier"
                allowed params:
                    - n_neighbors (int)
                    - weights ("uniform" | "distance")
                    - algorithm ("auto" | "ball_tree" | "kd_tree" | "brute")
                    - leaf_size (int)
                    - p (int)
                    - metric ("minkowski" | "euclidean" | "manhattan")
                    - n_jobs (int)


                ===============================================================================
                HYPERPARAMETER SELECTION GUIDELINES (DATASET SIZE AWARE)
                ===============================================================================
                Let N = number of rows.

                If N < 1 000 (SMALL):
                - Prefer stronger regularization
                - Smaller trees (max_depth 3–6)
                - Fewer estimators (100–300)
                - More conservative learning_rate (0.01–0.05)
                If 1 000 <= N <= 50 000 (MEDIUM):
                - Moderate regularization
                - max_depth 5–8
                - n_estimators 200–400
                - learning_rate 0.03–0.1
                If N > 50 000 (LARGE):
                - Can reduce regularization slightly
                - max_depth 6–10
                - n_estimators 300–500
                - learning_rate 0.05–0.15

                Always pick concrete numeric values consistent with these ranges.

                For logistic_regression specifically:
                - If N < 10_000: choose max_iter between 800 and 1500 (never below 800).
                - If 10_000 <= N <= 50_000: choose max_iter between 1500 and 2500.
                - If N > 50_000 or there are many features (> 100): choose max_iter between 2000 and 3000.
                Always set a concrete max_iter value in this range instead of a small default like 100 or 200.



                ===============================================================================
                FEATURE ENGINEERING RULES
                ===============================================================================
                encoding_strategy:
                - "none": if there are no categorical columns at all.
                - "one_hot": few categories per column (e.g. < 20) and moderate number of columns.
                - "label": for ordinal-like categoricals.
                - "frequency": for high-cardinality categoricals (many unique values).
                - "target": ONLY for regression with high-cardinality categoricals and enough rows.
                use_polynomial:
                - True only if numeric_continuous is small/moderate and N not extremely small.
                poly_degree:
                - 2 or 3 (never higher).
                use_interactions:
                - True if a few key numeric variables are important and N is not tiny.
                use_binning:
                - True when you want coarse grouping of numeric_continuous, often SMALL/MEDIUM N.
                bins:
                - Typically 5 or 10.
                bin_strategy:
                - "quantile" for roughly balanced bins, "uniform" for evenly spaced ranges.
                use_pca:
                - True only if there are many numeric features and you want dimensionality reduction.
                pca_n_components:
                - Reasonable range: 5–50 but must not exceed the number of numeric columns.
                SCALE_NUMERIC:
                - True: For regression tasks, gradient-based models
                - False: For tree-based only pipelines

                Your fe_config MUST be consistent with the dataset description.


                ===============================================================================
                FINAL AND MOST IMPORTANT RULES
                ===============================================================================
                - Return ONLY valid JSON.
                - Top-level keys MUST be exactly: "fe_config" and "model_list".
                - model_list MUST contain 3–5 enabled models.
                - model_name MUST be chosen from the allowed list for the given task.
                - params MUST use ONLY the allowed params for that model.
                - Do NOT include any free-form text or explanation anywhere.
                - Do NOT include trailing commas.

            """
        )
        # Determine dataset size category
        if n_rows < 1000:
            size_category = "SMALL"
        elif n_rows < 50000:
            size_category = "MEDIUM"
        else:
            size_category = "LARGE"
        user_msg = (
            f"""
            ═══════════════════════════════════════════════════════════════════════════════
            DATASET ANALYSIS
            ═══════════════════════════════════════════════════════════════════════════════

            BASIC INFO:
            - Rows: {n_rows} ({size_category} dataset)
            - Columns: {n_cols}
            - Task: {task.upper()}
            - Target: {target_column}

            COLUMN BREAKDOWN:
            - Numeric Continuous ({len(num_cont)}): {num_cont[:10]}{'...' if len(num_cont) > 10 else ''}
            - Numeric Categorical ({len(num_cat)}): {num_cat[:10]}{'...' if len(num_cat) > 10 else ''}
            - Categorical ({len(cat_cols)}): {cat_cols[:10]}{'...' if len(cat_cols) > 10 else ''}

            FULL SUMMARY:
            {json.dumps(cleaned_summary, indent=2)}

            ═══════════════════════════════════════════════════════════════════════════════
            YOUR TASK
            ═══════════════════════════════════════════════════════════════════════════════

            1. Choose optimal fe_config based on data characteristics
            2. Select 3-5 diverse models with SPECIFIC hyperparameters


            REMEMBER:
            - This is a {size_category} dataset, so adjust regularization accordingly
            - {"Use STRONGER regularization (higher reg_alpha, reg_lambda, lower max_depth)" if size_category == "SMALL" else "Use MODERATE regularization" if size_category == "MEDIUM" else "Can use LOWER regularization"}
            - DO NOT leave params empty - choose specific values!
            """
        )

        completion = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        content = completion.choices[0].message.content
        obj = json.loads(content)

        if "fe_config" in obj and isinstance(obj["fe_config"], dict):
                fe_config = obj["fe_config"]

        if "model_list" in obj and isinstance(obj["model_list"], list) and obj["model_list"]:
            raw_model_list = obj["model_list"]

    except Exception as e:
        fe_llm_error = str(e)
        
    agent_model_config = {"models": raw_model_list}
    model_list = build_model_list_from_config(agent_model_config, task)
    
    if task == "regression":
        allowed_models = {
            "linear_regression",
            "random_forest_regressor",
            "xgboost_regressor",
            "lightgbm_regressor",
            "gradient_boost_regressor",
            "catboost_regressor",
            "knn_regressor",
        }
    else:
        allowed_models = {
            "logistic_regression",
            "random_forest_classifier",
            "xgboost_classifier",
            "lightgbm_classifier",
            "gradient_boost_classifier",
            "catboost_classifier",
            "knn_classifier",
        }

    validated_list = [
        m for m in model_list
        if isinstance(m, dict) and m.get("model_name") in allowed_models
    ]

    # Only fallback if LLM provided *nothing*
    if not validated_list:
        print("⚠️ LLM produced invalid model_list, using defaults")
        validated_list = _default_model_list(task)

    model_list = validated_list

    fe_result = feature_engineering_agent.local(
        user_id=user_id,
        target_column=target_column,
        fe_config=fe_config,
    )
    if fe_result.get("status") != "success":
        return {
            "status": "error",
            "stage": "feature_engineering",
            "detail": fe_result,
            "version": version,
            "preprocess_config": preprocess_config,
            "fe_config": fe_config,
            "model_list": model_list,
            "preprocess_llm_error": preprocess_llm_error,
            "fe_llm_error": fe_llm_error,
            "viz_llm_error": viz_llm_error,
            "viz_plan_path": viz_plan_path,
            "dashboard_url_hint": f"/dashboard?user_id={user_id}&version={version}",
        }

    train_result = model_training_agent(
    user_id=user_id,
    target_column=target_column,
    model_list=model_list,
    )
    if train_result.get("status") != "success":
        return {
            "status": "error",
            "stage": "model_training",
            "detail": train_result,
            "version": version,
            "preprocess_config": preprocess_config,
            "fe_config": fe_config,
            "model_list": model_list,
            "preprocess_llm_error": preprocess_llm_error,
            "fe_llm_error": fe_llm_error,
            "viz_llm_error": viz_llm_error,
            "viz_plan_path": viz_plan_path,
            "dashboard_url_hint": f"/dashboard?user_id={user_id}&version={version}",
        }
        
        # ---- Human-readable decision report (why these configs/models) ----
    try:
        client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

        report_context = {
            "cleaned_summary": cleaned_summary,  
            "version": version,
            "task": task,
            "objective": objective,
            "target_column": target_column,
            "preprocess_config": preprocess_config,
            "fe_config": fe_config,
            "model_list": model_list,
            "training": train_result,
        }

        system_msg = """
        You are a SENIOR DATA SCIENTIST writing a clear, human-friendly report
        for a business user. You receive a JSON context describing:

        - cleaned_summary (rows, cols, column types, target info, missing_overview, target_distribution, is_imbalanced)
        - preprocess_config
        - fe_config (including imbalance handling choices)
        - model training results (train/test metrics, generalization gaps, best model)

        The JSON you get will look like:

        {
        "cleaned_summary": {
            "rows": int,
            "cols": int,
            "numeric_continuous": [...],
            "numeric_categorical": [...],
            "categorical_columns": [...],
            "target_column": "...",
            "task": "classification" | "regression",
            "objective": "...",
            "missing_overview": { "col": missing_ratio_float, ... },
            "target_distribution": {
            "counts": { "<class>": int, ... },
            "ratios": { "<class>": float, ... },
            "total": int
            } | null,
            "is_imbalanced": true | false | null
        },
        "preprocess_config": {...},
        "fe_config": {...},
        "model_list": [...],
        "training": {
            "results": [
            {
                "model_name": "...",
                "train_metrics": {...},
                "test_metrics": {...},
                "generalization_gap": float
            },
            ...
            ],
            "best_model_name": "..."
        }
        }

        You MUST output a concise MARKDOWN report with the following sections ONLY:

        ## 1. Data summary
        - 3–6 sentences.
        - ALWAYS use the numeric values from cleaned_summary:
        * Say: "The cleaned dataset contains **<rows> rows × <cols> columns**."
            where <rows> = cleaned_summary.rows and <cols> = cleaned_summary.cols.
            NEVER output the literal text "<rows>" or "<cols>" or "rows x cols".
        - Mention:
        * the target column name and whether the task is **classification** or **regression**
            using cleaned_summary.target_column and cleaned_summary.task;
        * whether there are significant missing values:
            - if cleaned_summary.missing_overview is empty or all near zero, say there are
            no significant missing-value issues;
            - otherwise briefly name 1–3 columns with notable missingness;
        * for classification, use cleaned_summary.target_distribution and
            cleaned_summary.is_imbalanced to say whether the target is **balanced**,
            **roughly balanced**, or **imbalanced**.
        - If cleaned_summary.target_distribution is null, simply say that detailed class
        distribution is not available instead of guessing.

        ## 2. Data preparation
        - 3–6 sentences.
        - Explain what was done at the cleaning stage using preprocess_config:
        * handling duplicates / ID index,
        * missing-data thresholds for columns and rows,
        * basic date/datetime handling (if enabled),
        * outlier handling strategy (e.g. IQR with a given multiplier).
        - Keep this high-level and readable for a non-technical user: focus on
        consistency, removing obviously bad values, and making the data ready for modelling.
        - Do NOT invent numeric thresholds that are not in preprocess_config. You may
        describe them qualitatively (e.g. “a standard IQR-based rule”).

        ## 3. Feature engineering & imbalance handling
        - 3–6 sentences.
        - Describe WHICH feature-engineering steps were requested in fe_config:
        * encoding strategy for categoricals (one-hot, label, ordinal, target, etc.),
        * scaling of numeric features (if enabled),
        * polynomial features or interactions (if enabled),
        * binning (if enabled),
        * PCA (if enabled).
        - For each major step, briefly explain WHY it is appropriate given:
        * the presence/absence of categorical columns,
        * whether the task is regression vs classification,
        * and whether the target is imbalanced.
        - Describe the imbalance strategy from fe_config:
        * If imbalance_strategy is not "none" (smote, smotenc, adasyn, smoteenn,
            smotetomek, random_over, random_under), explain in plain language that
            resampling was applied to reduce class imbalance and why that helps.
        * If imbalance_strategy is "none" but cleaned_summary.is_imbalanced is true,
            explain that the models rely on built-in handling such as class weights
            instead of resampling.
        - Do NOT talk about individual hyperparameters; keep the focus on data-level logic.

        ## 4. Models evaluated
        - Start with 1–2 sentences saying that several models were trained and evaluated
        on the same train/test split.
        - Then write a bullet list, ONE bullet per model in training.results.
        For each model:
        - use its model_name;
        - mention its test **accuracy** for classification or **RMSE** for regression,
            taking the value DIRECTLY from test_metrics;
        - describe whether the generalization_gap indicates the model is
            stable (small gap), mildly overfitting (moderate gap), or strongly
            overfitting (large gap);
        - briefly compare it to the others (e.g. “highest accuracy with low gap”,
            “slightly lower accuracy but very stable”).
        - Do NOT use tables in this section.

        ## 5. Data-focused improvements
        - 4–6 bullet points.
        - These bullets MUST focus ONLY on data-related improvements, NOT model tuning.
        Acceptable themes:
        * collecting more rows or more recent data,
        * improving data quality (fewer missing or noisy values),
        * adding or refining domain-specific features,
        * improving how categorical values or rare categories are recorded,
        * better handling of extreme outliers at the data-collection stage,
        * rebalancing the target through better sampling or business processes.
        - Do NOT suggest changing algorithms, hyperparameters, ensembles, or training
        procedures here.

        ## 6. Model metrics summary
        - Add a final section with a compact metrics table.
        - For CLASSIFICATION tasks, create this markdown table:

        | Model | Accuracy | Precision | Recall | F1 | Generalization gap |
        |-------|----------|-----------|--------|----|--------------------|

        - For REGRESSION tasks, create this markdown table:

        | Model | RMSE | MAE | R² | Generalization gap |
        |-------|------|-----|----|--------------------|

        - Fill each cell using the existing values from training.results:
        use each model’s test_metrics (accuracy, precision, recall, f1 for classification;
        rmse, mae, r2 for regression) and its generalization_gap.
        - If any metric is missing for a model, put `N/A` in that cell.
        - After the table you may add ONE short sentence naming which model is considered
        the best (using training.best_model_name) and why (e.g. “highest test accuracy
        with a small generalization gap”).
        - Do NOT output any “final answer”, LaTeX, or `\\boxed{}` expressions.

        GLOBAL RULES:
        - Use only information actually present in the JSON context.
        - Never speculate about columns, metrics, or transformations that are not mentioned.
        - Keep the tone professional but friendly and understandable by a non-expert.
        - Avoid internal implementation details (no mentions of “config dictionaries”,
        “JSON payloads”, or “pipelines”). Write as if you personally designed this
        workflow for the user.
        - Output MUST be Markdown only (no raw JSON, no code).
        """




        completion = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": json.dumps(report_context, indent=2)},
            ],
            temperature=0.2,
            max_tokens=800,
        )

        decision_report = completion.choices[0].message.content
        
        reports_folder = get_user_reports(user_id)
        os.makedirs(reports_folder, exist_ok=True)
        report_path = os.path.join(reports_folder, f"report_v{version}.md")
        with open(report_path, "w") as f:
            f.write(decision_report)

    except Exception:
        decision_report = None
        report_path = None


    return {
        "status": "success",
        "phase": "automl",
        "version": version,
        "task": task,
        "preprocess_config": preprocess_config,
        "fe_config": fe_config,
        "model_list": model_list,
        "training": train_result,
        "decision_report": decision_report,  # NEW
        "preprocess_llm_error": preprocess_llm_error,
        "fe_llm_error": fe_llm_error,
        "viz_llm_error": viz_llm_error,
        "viz_plan_path": viz_plan_path,
        "dashboard_url_hint": f"/dashboard?user_id={user_id}&version={version}",
        "decision_report_path": report_path,
    }


from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
import joblib
import json

web_app = FastAPI(
    title="AI Model Prediction API - Modal Deployment",
    description="Deployed ML model API on Modal",
    version="1.0.0"
)

# Add CORS middleware
web_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictionRequest(BaseModel):
    user_id: str
    version: int
    model_name: Optional[str] = None
    features: Dict[str, Any]

class PredictionResponse(BaseModel):
    prediction: float | str
    model_used: str
    confidence: Optional[float] = None
    processing_time_ms: float

@app.function(
    image=ai_image, 
    volumes={"/root/data": volume},
    timeout=300
)
def load_deployed_model(user_id: str, version: int, model_name: str):

    try:
        
        from pipeline.utils_paths import get_user_models, get_user_preprocessed, get_user_engineered
        
        models_folder = get_user_models(user_id)
        model_path = os.path.join(models_folder, f"model_v{version}_{model_name}.pkl")
        
        if not os.path.exists(model_path):
            return {"status": "error", "message": f"Model not found: {model_path}"}
        
        model = joblib.load(model_path)
        
        # Load preprocessing metadata
        pre_folder = get_user_preprocessed(user_id)
        metadata_path = os.path.join(pre_folder, f"metadata_v{version}.json")
        
        preprocessing_metadata = {}
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                preprocessing_metadata = json.load(f)
        
        # Load feature engineering config
        eng_folder = get_user_engineered(user_id)
        fe_config_path = os.path.join(eng_folder, f"fe_config_v{version}.json")
        
        fe_config = {}
        if os.path.exists(fe_config_path):
            with open(fe_config_path, "r") as f:
                fe_config = json.load(f)
        
        return {
            "status": "success",
            "model": model,
            "preprocessing_metadata": preprocessing_metadata,
            "fe_config": fe_config,
            "model_name": model_name,
            "user_id": user_id,
            "version": version
        }
    
    except Exception as e:
        return {"status": "error", "message": str(e)}
    
@app.function(
    image=ai_image, 
    volumes={"/root/data": volume},
    timeout=60
)
def make_prediction(
    user_id: str, 
    version: int, 
    model_name: Optional[str], 
    input_features: Dict[str, Any]
):
    """
    Make predictions with automatic best model selection.
    FIXED: Volume reload + target encoding + NaN handling.
    """
    import time
    import os
    import pandas as pd
    import numpy as np
    import re
    import json

    from pipeline.utils_paths import get_user_engineered, get_user_models
    from pipeline.Feature_Engineering import run_feature_engineering

    start_time = time.time()
    
    try:       
        # Auto-select best model if not specified
        if model_name is None:
            models_folder = get_user_models(user_id)
            
            # Check if folder exists
            if not os.path.exists(models_folder):
                return {
                    "status": "error",
                    "message": f"Models folder not found: {models_folder}. Run training first via /start endpoint."
                }
            
            model_files = [
                f for f in os.listdir(models_folder)
                if f.startswith(f"model_v{version}_") and f.endswith(".pkl")
            ]
            
            if not model_files:
                return {
                    "status": "error",
                    "message": f"No models found for version {version}. Available files: {os.listdir(models_folder)[:5]}"
                }
            
            best_score = float('inf')
            best_model_file = None
            
            for model_file in model_files:
                parts = model_file.replace(".pkl", "").split("_", 2)
                if len(parts) >= 3:
                    current_model_name = parts[2]
                    
                    metrics_file = os.path.join(
                        models_folder,
                        f"metrics_v{version}_{current_model_name}.json"
                    )
                    
                    if os.path.exists(metrics_file):
                        with open(metrics_file, 'r') as f:
                            metrics = json.load(f)
                            test_metrics = metrics.get("test", {})
                            
                            if "rmse" in test_metrics:
                                score = test_metrics["rmse"]
                                if score < best_score:
                                    best_score = score
                                    best_model_file = current_model_name
                            elif "accuracy" in test_metrics:
                                score = -test_metrics["accuracy"]
                                if score < best_score:
                                    best_score = score
                                    best_model_file = current_model_name
            
            if best_model_file is None:
                model_name = model_files[0].replace(".pkl", "").split("_", 2)[2]
            else:
                model_name = best_model_file
            
            print(f"✨ Auto-selected best model: {model_name}")
        
        # Load model + metadata
        model_data = load_deployed_model.local(user_id, version, model_name)
        
        if model_data["status"] != "success":
            return {
                "status": "error",
                "message": model_data.get("message", "Failed to load model")
            }
        
        model = model_data["model"]
        preprocessing_metadata = model_data.get("preprocessing_metadata", {}) or {}
        fe_config = model_data.get("fe_config", {}) or {}

        target_column = preprocessing_metadata.get("target_column", "price")

        eng_folder = get_user_engineered(user_id)
        engineered_path = os.path.join(eng_folder, f"engineered_v{version}.parquet")
        if not os.path.exists(engineered_path):
            return {
                "status": "error",
                "message": f"Engineered dataset not found: {engineered_path}"
            }

        train_df = pd.read_parquet(engineered_path)
        train_feature_cols = [c for c in train_df.columns if c != target_column]

        # Prepare raw input
        df_raw = pd.DataFrame([input_features])

        # Handle missing values
        for col in df_raw.columns:
            if df_raw[col].dtype.kind in "biufc":
                df_raw[col] = pd.to_numeric(df_raw[col], errors="coerce")
                df_raw[col].fillna(0, inplace=True)
            else:
                df_raw[col] = df_raw[col].astype(str).fillna("unknown")

        # Add dummy target
        df_fe_input = df_raw.copy()
        if target_column not in df_fe_input.columns:
            df_fe_input[target_column] = 0

        # Run feature engineering
        fe_config_infer = dict(fe_config or {})
        fe_config_infer["imbalance_strategy"] = "none"
        
        # ✨ FIX: Disable target encoding for single-row predictions
        if fe_config_infer.get("encoding_strategy") == "target":
            fe_config_infer["encoding_strategy"] = "one_hot"
        
        engineered_new = run_feature_engineering(
            df=df_fe_input,
            target_column=target_column,
            fe_config=fe_config_infer,
        )

        if target_column in engineered_new.columns:
            engineered_new = engineered_new.drop(columns=[target_column])

        # Sanitize column names
        def _sanitize_column_name(name: str) -> str:
            name = name.replace(" ", "_")
            name = re.sub(r"[^0-9A-Za-z_]+", "_", name)
            name = re.sub(r"_+", "_", name).strip("_")
            return name[:80]
        
        engineered_new.columns = [_sanitize_column_name(col) for col in engineered_new.columns]

        # Align with training features
        X = engineered_new.reindex(columns=train_feature_cols, fill_value=0)

        # Final NaN check
        if X.isnull().any().any():
            X = X.fillna(0)

        # Predict
        prediction = model.predict(X)[0]

        confidence = None
        if hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(X)[0]
                confidence = float(np.max(proba))
            except Exception:
                confidence = None
        
        processing_time = (time.time() - start_time) * 1000.0
        
        return {
            "status": "success",
            "prediction": float(prediction) if isinstance(prediction, (int, float, np.number)) else str(prediction),
            "model_used": model_name,
            "confidence": confidence,
            "processing_time_ms": processing_time,
        }
    
    except Exception as e:
        import traceback
        return {
            "status": "error",
            "message": f"Prediction failed: {str(e)}",
            "traceback": traceback.format_exc()
        }

@app.function(
    image=ai_image, 
    volumes={"/root/data": volume},
    timeout=300
)
def batch_predict(
    user_id: str, 
    version: int, 
    model_name: str, 
    batch_data: list[Dict[str, Any]]
):

    results = []
    
    for item in batch_data:
        result = make_prediction.local(user_id, version, model_name, item)
        results.append({
            "input": item,
            **result
        })
    
    return {
        "status": "success",
        "predictions": results,
        "count": len(results),
        "model_name": model_name,
        "user_id": user_id,
        "version": version
    }

@app.function(
    image=ai_image,
    volumes={"/root/data": volume}
)
@web_app.post("/predict")
def api_predict(request: PredictionRequest):

    result = make_prediction.local(
        user_id=request.user_id,
        version=request.version,
        model_name=request.model_name,
        input_features=request.features
    )
    
    if result["status"] != "success":
        raise HTTPException(status_code=500, detail=result.get("message", "Prediction failed"))
    
    return PredictionResponse(
        prediction=result["prediction"],
        model_used=result["model_used"],
        confidence=result.get("confidence"),
        processing_time_ms=result["processing_time_ms"]
    )
@web_app.get("/schema/{user_id}/{version}")
def api_schema(user_id: str, version: int):
    """
    Return cleaned columns + logical input type for a given user/version.

    Rules:
    - Exclude any column whose name contains 'id' (case-insensitive).
    - raw_labels are ONLY included for:
        * true categorical columns (object-like, in categorical_columns),
        * non-numeric target columns.
    - No raw_labels for numeric columns (even if low-cardinality).
    """
    from pipeline.utils_paths import get_user_preprocessed

    pre_folder = get_user_preprocessed(user_id)
    cleaned_path = os.path.join(pre_folder, f"cleaned_v{version}.parquet")
    metadata_path = os.path.join(pre_folder, f"metadata_v{version}.json")

    if not os.path.exists(cleaned_path) or not os.path.exists(metadata_path):
        raise HTTPException(
            status_code=404,
            detail="Cleaned data or metadata not found for this version.",
        )

    df = pd.read_parquet(cleaned_path)
    with open(metadata_path, "r") as f:
        meta = json.load(f)

    num_cont = set(meta.get("numeric_continuous", []))
    num_cat = set(meta.get("numeric_categorical", []))
    cat_cols = set(meta.get("categorical_columns", []))
    date_cols = set(meta.get("date_columns", []))
    target_col = meta.get("target_column")

    schema = []
    MAX_LABELS = 200  # safety cap for huge categoricals (optional)

    for col, dt in df.dtypes.items():
        col_lower = col.lower()

        # 1) Skip any column containing 'id'
        if "id" in col_lower:
            continue

        dt_str = str(dt)

        # 2) Determine logical input_type
        if col == target_col:
            input_type = "target"
        elif col in num_cont or col in num_cat:
            input_type = "numeric"
        elif col in date_cols:
            input_type = "datetime"
        elif col in cat_cols:
            input_type = "categorical"
        else:
            # Fallback if column not in metadata lists
            if np.issubdtype(dt, np.number):
                input_type = "numeric"
            elif np.issubdtype(dt, np.datetime64):
                input_type = "datetime"
            else:
                input_type = "string"

        # 3) raw_labels ONLY for true categoricals (non-numeric)
        raw_labels = None
        is_true_categorical = col in cat_cols
        is_categorical_target = (
            col == target_col and not np.issubdtype(dt, np.number)
        )

        if is_true_categorical or is_categorical_target:
            values = df[col].dropna().unique().tolist()
            if len(values) > MAX_LABELS:
                values = values[:MAX_LABELS]
            raw_labels = values

        schema.append(
            {
                "name": col,
                "pandas_dtype": dt_str,
                "input_type": input_type,
                "raw_labels": raw_labels,  # always null for numeric / datetime / string
            }
        )

    return {
        "user_id": user_id,
        "version": version,
        "target_column": target_col,
        "schema": schema,
    }


@web_app.get("/models/{user_id}/{version}")
def api_models_summary(user_id: str, version: int):
    """
    Return best model + all models metrics + markdown report for a given version.
    """
    from pipeline.utils_paths import (
        get_user_models,
        get_user_reports,
        get_user_preprocessed,
    )

    models_folder = get_user_models(user_id)
    if not os.path.exists(models_folder):
        raise HTTPException(
            status_code=404,
            detail="Models folder not found for this user.",
        )

    # Load cleaned data + metadata to infer task & target
    pre_folder = get_user_preprocessed(user_id)
    cleaned_path = os.path.join(pre_folder, f"cleaned_v{version}.parquet")
    metadata_path = os.path.join(pre_folder, f"metadata_v{version}.json")

    if not os.path.exists(cleaned_path) or not os.path.exists(metadata_path):
        raise HTTPException(
            status_code=404,
            detail="Cleaned data or metadata not found for this version.",
        )

    df = pd.read_parquet(cleaned_path)
    with open(metadata_path, "r") as f:
        meta = json.load(f)

    target_col = meta.get("target_column")
    if target_col is None or target_col not in df.columns:
        raise HTTPException(
            status_code=400,
            detail="Target column not found in cleaned data.",
        )

    y = df[target_col]
    is_numeric = np.issubdtype(y.dtype, np.number)
    n_unique = int(y.nunique(dropna=True))

    # Same heuristic you already use elsewhere
    if is_numeric and n_unique <= 10:
        task_type = "classification"
    else:
        task_type = "regression" if is_numeric else "classification"

    # Collect metrics for THIS version only
    metrics_files = [
        f
        for f in os.listdir(models_folder)
        if f.startswith(f"metrics_v{version}_") and f.endswith(".json")
    ]
    if not metrics_files:
        raise HTTPException(
            status_code=404,
            detail="No metrics files found for this version.",
        )

    all_models = []
    for fname in metrics_files:
        path = os.path.join(models_folder, fname)
        with open(path, "r") as f:
            data = json.load(f)

        # "metrics_v{version}_{model_name}.json" → pull the model_name part
        model_name = fname.replace(".json", "").split("_", 2)[2]

        train_m = data.get("train", {}) or {}
        test_m = data.get("test", {}) or {}
        gap = None

        if task_type == "regression" and "rmse" in test_m and "rmse" in train_m:
            gap = test_m["rmse"] - train_m["rmse"]
        elif task_type == "classification" and "accuracy" in test_m and "accuracy" in train_m:
            gap = train_m["accuracy"] - test_m["accuracy"]

        all_models.append(
            {
                "name": model_name,
                "train_metrics": train_m,
                "test_metrics": test_m,
                "generalization_gap": gap,
            }
        )

    # Pick best model using same idea as in /start:
    best_model = None
    best_score = None

    for m in all_models:
        test_m = m["test_metrics"]

        if task_type == "regression" and "rmse" in test_m:
            score = test_m["rmse"]  # lower is better
            better = best_score is None or score < best_score
        elif task_type == "classification" and "accuracy" in test_m:
            score = test_m["accuracy"]  # higher is better
            better = best_score is None or score > best_score
        else:
            continue

        if better:
            best_score = score
            best_model = m

    # Load markdown report if it exists
    reports_folder = get_user_reports(user_id)
    report_path = os.path.join(reports_folder, f"report_v{version}.md")
    report_md = None
    if os.path.exists(report_path):
        with open(report_path, "r") as f:
            report_md = f.read()

    return {
        "user_id": user_id,
        "version": version,
        "task": task_type,
        "target_column": target_col,
        "best_model": best_model,
        "all_models": all_models,
        "report_markdown": report_md,
    }
@web_app.get("/dashboard/{user_id}/{version}")
def api_dashboard(user_id: str, version: int):
    """
    Return only the ready-to-render charts list for a given version.
    """
    from pipeline.utils_paths import get_user_visualizations, get_user_dashboard_json

    viz_folder = get_user_visualizations(user_id)
    versioned_path = os.path.join(viz_folder, f"dashboard_data_v{version}.json")
    latest_path = get_user_dashboard_json(user_id)

    # Prefer versioned data if it exists; fall back to latest
    if os.path.exists(versioned_path):
        dash_path = versioned_path
    elif os.path.exists(latest_path):
        dash_path = latest_path
    else:
        raise HTTPException(
            status_code=404,
            detail="Dashboard data not found for this user/version.",
        )

    with open(dash_path, "r") as f:
        dashboard_data = json.load(f)

    # Only expose charts to the frontend
    charts = dashboard_data.get("charts", [])

    return {
        "user_id": user_id,
        "version": version,
        "charts": charts,
    }


@app.function(
    image=ai_image,
    volumes={"/root/data": volume},
    timeout=600,
    container_idle_timeout=300,
    secrets=[groq_secret],
)
@modal.asgi_app()
def fastapi_app():

    return web_app

@app.function(
    image=ai_image,
    volumes={"/root/data": volume},
    timeout=300
)
def deploy_model(
    user_id: str, 
    version: int, 
    model_name: str,
    deployment_name: Optional[str] = None
):

    try:
        from pipeline.utils_paths import get_user_models
        
        models_folder = get_user_models(user_id)
        model_path = os.path.join(models_folder, f"model_v{version}_{model_name}.pkl") 

        if not os.path.exists(model_path):
            return {
                "status": "error",
                "message": f"Model not found: {model_name}"
            }
        
        if not deployment_name:
            deployment_name = f"{user_id}_{model_name}_v{version}"
        
        deployment_info = {
            "deployment_name": deployment_name,
            "user_id": user_id,
            "version": version,
            "model_name": model_name,
            "deployed_at": pd.Timestamp.now().isoformat(),
            "status": "active"
        }
        
        deployment_folder = os.path.join("/root/data", "deployments", user_id)
        os.makedirs(deployment_folder, exist_ok=True)
        
        deployment_file = os.path.join(deployment_folder, f"{deployment_name}.json")
        with open(deployment_file, "w") as f:
            json.dump(deployment_info, f, indent=2)
        
        deployment_url = "https://omarnashat2004--agentic-ai-app-fastapi-app-dev.modal.run"

        
        return {
            "status": "success",
            "deployment_name": deployment_name,
            "deployment_url": deployment_url,
            "api_endpoints": {
                "predict": f"{deployment_url}/predict",
                "health": f"{deployment_url}/health",
                "docs": f"{deployment_url}/docs"
            },
            "deployment_info": deployment_info,
            "message": f"Model {model_name} deployed successfully",
            "usage_example": {
                "endpoint": f"{deployment_url}/predict",
                "method": "POST",
                "body": {
                    "user_id": user_id,
                    "version": version,
                    "model_name": model_name,
                    "features": {
                        "feature1": "value1",
                        "feature2": "value2"
                    }
                }
            }
        }
    
    except Exception as e:
        return {
            "status": "error",
            "message": f"Deployment failed: {str(e)}"
        }

@app.function(
    image=ai_image,
    volumes={"/root/data": volume}
)
def list_deployments(user_id: str):

    try:
        deployment_folder = os.path.join("/root/data", "deployments", user_id)
        
        if not os.path.exists(deployment_folder):
            return {
                "status": "success",
                "deployments": [],
                "count": 0
            }
        
        deployments = []
        for file in os.listdir(deployment_folder):
            if file.endswith(".json"):
                with open(os.path.join(deployment_folder, file), "r") as f:
                    deployment = json.load(f)
                    deployments.append(deployment)
        
        return {
            "status": "success",
            "deployments": deployments,
            "count": len(deployments)
        }
    
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to list deployments: {str(e)}"
        }
        
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
from fastapi import HTTPException

class StartRequest(BaseModel):
    user_id: str = Field(..., description="Logical user id for per-user isolation.")
    filename: str = Field(..., description="Original filename of the uploaded CSV.")
    url: str = Field(..., description="Path/URL where Modal can read the uploaded file.")
    target_column: str = Field(..., description="Target column selected in the UI.")
    run_automl: bool = Field(
        False,
        description="Checkbox from the UI. False = EDA only, True = full AutoML.",
    )
    objective: str = Field("auto", description="AutoML objective (usually 'auto').")


@app.function(
    image=ai_image,
    volumes={"/root/data": volume},
    timeout=900,
    secrets=[groq_secret],
)
@web_app.post("/start")
def api_start(request: StartRequest):
    """
    Orchestrate ingestion + Auto-AI, but return only:
    - basic dataset summary (rows, cols, column names)
    - task + mode (eda vs automl)
    - version
    - endpoint URLs for schema / models / dashboard (+ predict if automl)
    """

    # 1) Ingestion ------------------------------------------------------
    ingest_result = ingestion_agent.local(
        user_id=request.user_id,
        filename=request.filename,
        url=request.url,
    )

    if ingest_result.get("status") != "success":
        raise HTTPException(status_code=500, detail=ingest_result)

    ingested_version = ingest_result.get("version") or 1

    # 2) Auto-AI (preprocessing + viz + optional AutoML) ---------------
    auto_result = auto_ai_agent.local(
        user_id=request.user_id,
        target_column=request.target_column,
        objective=request.objective,
        run_automl=request.run_automl,
    )

    if auto_result.get("status") != "success":
        # Still bubble up the structured error, but no huge payload on success.
        raise HTTPException(status_code=500, detail=auto_result)

    # Version from auto_ai_agent if present, fall back to ingestion version
    final_version = auto_result.get("version") or ingested_version

    # 3) Lightweight dataset summary (from ingestion only) -------------
    n_rows = ingest_result.get("shape", [None, None])[0]
    n_cols = ingest_result.get("shape", [None, None])[1]
    column_names = ingest_result.get("columns", []) or []

    dataset_summary = {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "columns": column_names,
    }

    # 4) Mode / task ----------------------------------------------------
    mode = "automl" if request.run_automl else "eda"
    task_type = auto_result.get("task")  # "classification" / "regression"

    # 5) Endpoint hints for frontend wiring -----------------------------
    # These match your existing GET routes:
    #   /schema/{user_id}/{version}
    #   /models/{user_id}/{version}
    #   /dashboard/{user_id}/{version}
    endpoints = {
        "schema": f"/schema/{request.user_id}/{final_version}",
        "dashboard": f"/dashboard/{request.user_id}/{final_version}",
        # Only meaningful if AutoML was run:
        "models": (
            f"/models/{request.user_id}/{final_version}"
            if request.run_automl
            else None
        ),
        # Predict endpoint is global; backend will POST here with user_id + version
        "predict": "/predict" if request.run_automl else None,
    }

    # 6) Final minimal response ----------------------------------------
    return {
        "status": "success",
        "mode": mode,
        "user_id": request.user_id,
        "version": final_version,
        "task": task_type,
        "endpoints": endpoints,
    }


