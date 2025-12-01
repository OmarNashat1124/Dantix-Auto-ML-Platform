import os
import pandas as pd
import numpy as np
import logging
import re
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder, StandardScaler

from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from pipeline.utils_paths import get_user_engineered
from pipeline.Preprocessing import detect_numeric_and_categorical
from pandas.api.types import is_datetime64_any_dtype

from imblearn.over_sampling import (
    SMOTE,
    SMOTENC,
    RandomOverSampler,
    ADASYN,
)
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN, SMOTETomek

logger = logging.getLogger(__name__)

def _sanitize_column_name(name: str) -> str:
    """
    Sanitize column names to be compatible with all ML libraries.
    This MUST match the sanitization in model_training_agent.
    """
    name = str(name)
    name = name.replace(" ", "_")
    name = re.sub(r"[^0-9A-Za-z_]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name[:80]
def expand_datetime_features(df: pd.DataFrame, target_column: str | None = None) -> pd.DataFrame:
    df = df.copy()

    # 🔧 FIX: detect datetime columns without using a fragile dtype string
    date_cols = [
        col for col in df.columns
        if is_datetime64_any_dtype(df[col])
    ]

    if not date_cols:
        return df

    logger.info(f"Expanding datetime columns into numeric features: {date_cols}")

    for col in date_cols:
        if col == target_column:
            continue

        df[f"{col}_year"] = df[col].dt.year
        df[f"{col}_month"] = df[col].dt.month
        df[f"{col}_day"] = df[col].dt.day
        df[f"{col}_dayofweek"] = df[col].dt.dayofweek

    drop_cols = [c for c in date_cols if c != target_column]
    if drop_cols:
        df = df.drop(columns=drop_cols)
        logger.info(f"Dropped original datetime columns after expansion: {drop_cols}")

    return df



def one_hot_encode(df, categorical_columns):
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    encoded = encoder.fit_transform(df[categorical_columns])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categorical_columns), index=df.index)
    df = df.drop(columns=categorical_columns)
    df = pd.concat([df, encoded_df], axis=1)
    
    # ✨ CRITICAL: Sanitize column names after one-hot encoding
    df.columns = [_sanitize_column_name(col) for col in df.columns]
    
    logger.info(f"Applied One-Hot Encoding to {len(categorical_columns)} columns. New shape: {df.shape}.")
    return df

def label_encode(df, categorical_columns):
    df = df.copy()
    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    logger.info(f"Applied Label Encoding to {len(categorical_columns)} columns.")
    return df

def ordinal_encode(df, categorical_columns):
    encoder = OrdinalEncoder()
    df[categorical_columns] = encoder.fit_transform(df[categorical_columns].astype(str))
    logger.info(f"Applied Ordinal Encoding to {len(categorical_columns)} columns.")
    return df

def frequency_encode(df, categorical_columns):
    df = df.copy()
    for col in categorical_columns:
        freqs = df[col].value_counts()
        df[col] = df[col].map(freqs)
    logger.info(f"Applied Frequency Encoding to {len(categorical_columns)} columns.")
    return df

def target_encode(df, categorical_columns, target):
    """Target encoding with K-Fold - FIXED for single-row predictions."""
    df = df.copy()
    
    # ✨ FIX: Handle single-row predictions
    if len(df) < 5:
        logger.warning(
            f"Target encoding needs 5+ rows, got {len(df)}. Using fallback encoding."
        )
        for col in categorical_columns:
            df[col + "_te"] = 0  # Safe default
        df = df.drop(columns=categorical_columns)
        df.columns = [_sanitize_column_name(col) for col in df.columns]
        return df
    
    # Original K-Fold logic for training
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    global_mean = df[target].mean()
    for col in categorical_columns:
        df[col + "_te"] = 0
        for train_idx, val_idx in kfold.split(df):
            train_fold, val_fold = df.iloc[train_idx], df.iloc[val_idx]
            means = train_fold.groupby(col)[target].mean()
            df.loc[df.index[val_idx], col + "_te"] = df.loc[df.index[val_idx], col].map(means)
        df[col + "_te"] = df[col + "_te"].fillna(global_mean)
    df = df.drop(columns=categorical_columns)
    df.columns = [_sanitize_column_name(col) for col in df.columns]
    logger.info(f"Applied Target Encoding to {len(categorical_columns)} columns.")
    return df

def polynomial_features(df, numeric_columns, degree=2):
    df = df.copy()
    for col in numeric_columns:
        df[col + "_sq"] = df[col] ** 2
        if degree >= 3:
            df[col + "_cube"] = df[col] ** 3
    
    # ✨ Sanitize new column names
    df.columns = [_sanitize_column_name(col) for col in df.columns]
    
    logger.info(f"Applied Polynomial Features (degree={degree}) to {len(numeric_columns)} numeric columns.")
    return df

def numeric_interactions(df, numeric_columns):
    df = df.copy()
    count = 0
    for i in range(len(numeric_columns)):
        for j in range(i+1, len(numeric_columns)):
            a, b = numeric_columns[i], numeric_columns[j]
            df[f"{a}_x_{b}"] = df[a] * df[b]
            count += 1
    
    # ✨ Sanitize new column names
    df.columns = [_sanitize_column_name(col) for col in df.columns]
    
    logger.info(f"Generated {count} numeric interaction features.")
    return df

def bin_numeric(df, numeric_columns, bins=5, strategy="quantile"):
    df = df.copy()
    for col in numeric_columns:
        if strategy == "uniform":
            df[col + "_bin"] = pd.cut(df[col], bins=bins, labels=False)
        else:
            df[col + "_bin"] = pd.qcut(df[col], q=bins, labels=False, duplicates="drop")
    
    # ✨ Sanitize new column names
    df.columns = [_sanitize_column_name(col) for col in df.columns]
    
    logger.info(f"Applied binning to {len(numeric_columns)} numeric columns using '{strategy}' strategy.")
    return df

def apply_pca(df, n_components=10):
    pca = PCA(n_components=n_components)
    arr = pca.fit_transform(df)
    new_cols = [f"pca_{i+1}" for i in range(n_components)]
    logger.info(f"Applied PCA reducing to {n_components} components.")
    return pd.DataFrame(arr, columns=new_cols, index=df.index)


def save_engineered(df, user_id: str):
    folder = get_user_engineered(user_id)
    path = os.path.join(folder, "engineered.parquet")
    df.to_parquet(path, index=False)
    return path

def handle_target_imbalance(
    df: pd.DataFrame,
    target_column: str,
    strategy: str = "none",
    sampling_strategy="auto",
    smote_k_neighbors: int = 5,
    random_state: int = 42,
    minority_threshold: float = 0.2,
) -> pd.DataFrame:
    """
    Handle target imbalance by resampling the data.

    strategy:
        "none",
        "random_over",
        "random_under",
        "smote",
        "smotenc",
        "adasyn",
        "smoteenn",
        "smotetomek"
    """
    df = df.copy()

    if strategy in (None, "", "none"):
        logger.info("Imbalance strategy 'none' - skipping resampling.")
        return df

    if target_column not in df.columns:
        logger.warning(
            "Target column '%s' not found. Skipping imbalance handling.",
            target_column,
        )
        return df

    y = df[target_column]
    value_counts = y.value_counts()
    if len(value_counts) < 2:
        logger.info("Target has <2 classes. Skipping imbalance handling.")
        return df

    ratios = value_counts / len(y)
    minority_ratio = float(ratios.min())

    if minority_ratio >= minority_threshold:
        logger.info(
            "Target not considered imbalanced (minority_ratio=%.4f, threshold=%.4f). "
            "Skipping resampling.",
            minority_ratio,
            minority_threshold,
        )
        return df



    # --- NEW: guard against too-few minority samples for SMOTE-like methods ---
    minority_count = int(value_counts.min())
    if strategy in {"smote", "smotenc", "adasyn", "smoteenn", "smotetomek"}:
        if minority_count <= 2:
            logger.warning(
                "Too few samples in the minority class (%d) for SMOTE-style '%s'. "
                "Skipping resampling for safety.",
                minority_count,
                strategy,
            )
            return df

        # SMOTE requires k_neighbors < n_samples_minority
        safe_k = max(1, min(smote_k_neighbors, minority_count - 1))
    else:
        safe_k = smote_k_neighbors  # not used, but keep for interface

    X = df.drop(columns=[target_column])

    # Use existing helper to understand column types
    from pipeline.Preprocessing import detect_numeric_and_categorical

    numeric_continuous, numeric_categorical, categorical = detect_numeric_and_categorical(
        df, target_column=target_column
    )

    cat_for_smote = [c for c in (categorical + numeric_categorical) if c in X.columns]
    categorical_indices = [X.columns.get_loc(c) for c in cat_for_smote]

    # Encode categoricals as integer codes
    if cat_for_smote:
        X = X.copy()
        for col in cat_for_smote:
            X[col] = X[col].astype("category").cat.codes

    # ---------------- STRATEGY BRANCHES ----------------
    if strategy == "random_over":
        sampler = RandomOverSampler(
            sampling_strategy=sampling_strategy,
            random_state=random_state,
        )
        X_resampled, y_resampled = sampler.fit_resample(X, y)

    elif strategy == "random_under":
        sampler = RandomUnderSampler(
            sampling_strategy=sampling_strategy,
            random_state=random_state,
        )
        X_resampled, y_resampled = sampler.fit_resample(X, y)

    elif strategy in {"smote", "smotenc"}:
        if categorical_indices:
            sampler = SMOTENC(
                categorical_features=categorical_indices,
                sampling_strategy=sampling_strategy,
                k_neighbors=safe_k,
                random_state=random_state,
            )
        else:
            sampler = SMOTE(
                sampling_strategy=sampling_strategy,
                k_neighbors=safe_k,
                random_state=random_state,
            )
        X_resampled, y_resampled = sampler.fit_resample(X, y)

    elif strategy == "adasyn":
        sampler = ADASYN(
            sampling_strategy=sampling_strategy,
            n_neighbors=safe_k,
            random_state=random_state,
        )
        X_resampled, y_resampled = sampler.fit_resample(X, y)

    elif strategy == "smoteenn":
        # SMOTEENN uses an internal SMOTE; pass our safe_k via the smote argument
        base_smote = SMOTE(
            sampling_strategy=sampling_strategy,
            k_neighbors=safe_k,
            random_state=random_state,
        )
        sampler = SMOTEENN(
            smote=base_smote,
            sampling_strategy=sampling_strategy,
            random_state=random_state,
        )
        X_resampled, y_resampled = sampler.fit_resample(X, y)

    elif strategy == "smotetomek":
        base_smote = SMOTE(
            sampling_strategy=sampling_strategy,
            k_neighbors=safe_k,
            random_state=random_state,
        )
        sampler = SMOTETomek(
            smote=base_smote,
            sampling_strategy=sampling_strategy,
            random_state=random_state,
        )
        X_resampled, y_resampled = sampler.fit_resample(X, y)

    else:
        logger.warning(
            "Unknown imbalance strategy '%s'. Skipping resampling.", strategy
        )
        return df

    X_resampled = pd.DataFrame(X_resampled, columns=X.columns)
    y_resampled = pd.Series(y_resampled, name=target_column)
    df_resampled = pd.concat([X_resampled, y_resampled], axis=1)
    df_resampled = df_resampled[df.columns]

    logger.info(
        "Applied imbalance strategy '%s'. Original samples: %d, new samples: %d.",
        strategy,
        len(df),
        len(df_resampled),
    )
    return df_resampled


def scale_numeric_features(df: pd.DataFrame, target_column: str | None = None) -> pd.DataFrame:
    """
    Scale all numeric feature columns (excluding the target) using StandardScaler.
    This is called at the END of feature engineering so visualizations
    stay on unscaled cleaned data, but models see scaled features.
    """
    numeric_cols = df.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()

    if target_column in numeric_cols:
        numeric_cols.remove(target_column)

    if not numeric_cols:
        logger.info("No numeric feature columns to scale in feature engineering.")
        return df

    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    logger.info(f"Scaled numeric feature columns: {numeric_cols}")
    return df

def run_feature_engineering(
    df: pd.DataFrame,
    target_column: str,
    fe_config: dict | None = None,
) -> pd.DataFrame:


    default_config = {
        "encoding_strategy": "none",
        "use_polynomial": False,
        "poly_degree": 2,
        "use_interactions": False,
        "use_binning": False,
        "bins": 5,
        "bin_strategy": "quantile",
        "use_pca": False,
        "pca_n_components": 10,
        "scale_numeric": True, 
        "imbalance_strategy": "none",
        "imbalance_sampling_strategy": "auto",
        "imbalance_smote_k_neighbors": 5,
        "imbalance_random_state": 42,
        "imbalance_minority_threshold": 0.2,  
    }

    if fe_config is None:
        fe_config = {}
    cfg = {**default_config, **fe_config}


    df = expand_datetime_features(df, target_column=target_column)
    
    numeric_continuous, numeric_categorical, categorical = detect_numeric_and_categorical(
        df, target_column=target_column
    )
    # Always drop ID-like columns before encoding
    df = df.drop(columns=[c for c in df.columns if "id" in c.lower()], errors="ignore")

    encoding = cfg["encoding_strategy"]

    if encoding == "one_hot":
        cat_for_encoding = categorical + numeric_categorical
        if cat_for_encoding:
            df = one_hot_encode(df, cat_for_encoding)

    elif encoding == "label" and categorical:
        df = label_encode(df, categorical)

    elif encoding == "ordinal" and categorical:
        df = ordinal_encode(df, categorical)

    elif encoding == "frequency" and categorical:
        df = frequency_encode(df, categorical)

    elif encoding == "target" and categorical and target_column in df.columns:
        df = target_encode(df, categorical, target_column)

    if cfg["use_polynomial"] and numeric_continuous:
        df = polynomial_features(df, numeric_continuous, degree=cfg["poly_degree"])

    if cfg["use_interactions"] and numeric_continuous:
        df = numeric_interactions(df, numeric_continuous)

    if cfg["use_binning"] and numeric_continuous:
        df = bin_numeric(
            df,
            numeric_continuous,
            bins=cfg["bins"],
            strategy=cfg["bin_strategy"],
        )

    if cfg["use_pca"]:
        feature_cols = [c for c in df.columns if c != target_column]
        numeric_for_pca = df[feature_cols].select_dtypes(
            include=["int64", "float64", "int32", "float32"]
        )

        if not numeric_for_pca.empty:
            n_comp = min(cfg["pca_n_components"], numeric_for_pca.shape[1])
            pca_df = apply_pca(numeric_for_pca, n_components=n_comp)

            remaining_cols = [c for c in df.columns if c not in numeric_for_pca.columns]
            df = pd.concat([df[remaining_cols], pca_df], axis=1)
            
    if cfg.get("scale_numeric", False):
        df = scale_numeric_features(df, target_column=target_column)
    
    # ✨ FINAL: Sanitize ALL column names at the end
    df.columns = [_sanitize_column_name(col) for col in df.columns]
    
    logger.info(
        f"run_feature_engineering completed. Final shape: {df.shape[0]} rows × {df.shape[1]} cols"
    )
    return df