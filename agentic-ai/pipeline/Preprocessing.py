import os
import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import StandardScaler
from datetime import datetime
from pandas.api.types import is_integer_dtype 
logger = logging.getLogger(__name__)

def set_id_columns_as_index(df: pd.DataFrame):
    logger.info("Checking for ID-like columns...")
    id_cols = [col for col in df.columns if "id" in col.lower()]
    logger.info(f"Identified ID-like columns: {id_cols}")

    if len(id_cols) == 1:
        df = df.set_index(id_cols[0])
        logger.info(f"Set '{id_cols[0]}' as index column.")
    elif len(id_cols) > 1:
        df = df.set_index(id_cols)
        logger.info(f"Set multiple ID columns as index: {id_cols}")
    else:
        logger.info("No ID-like columns found to set as index.")

    return df

def remove_duplicates(df):
    logger.info("Removing duplicate rows...")
    before = len(df)
    df = df.drop_duplicates()
    removed = before - len(df)
    logger.info(f"Removed {removed} duplicates.")
    return df


def handle_missing_data(df, col_threshold=0.5, row_threshold=0.05, target_column=None):
    logger.info("Handling missing data...")

    before_cols = df.shape[1]
    before_rows = df.shape[0]
    logger.info(f"Initial shape before missing-data handling: {before_rows} rows × {before_cols} cols")

    columns_to_check = [c for c in df.columns if c != target_column]
    keep_cols = [c for c in columns_to_check if df[c].isnull().mean() < col_threshold]

    if target_column in df.columns:
        keep_cols.append(target_column)

    removed_columns = before_cols - len(keep_cols)
    logger.info(f"Columns removed due to col-threshold ({col_threshold}): {removed_columns}")

    df = df[keep_cols]

    before_rows_2 = len(df)
    df = df[df.isnull().mean(axis=1) < row_threshold]
    removed_rows = before_rows_2 - len(df)
    logger.info(f"Rows removed due to row-threshold ({row_threshold}): {removed_rows}")

    logger.info(f"Shape after missing-data handling: {df.shape[0]} rows × {df.shape[1]} cols")

    return df


def handle_date_columns(df, target_column=None):
    logger.info("Handling date/time-like columns...")
    now = datetime.now()

    for col in df.columns:
        if col == target_column:
            continue

        if "date" in col.lower() or "time" in col.lower():
            logger.info(f"Attempting datetime conversion for '{col}'")
            converted = pd.to_datetime(df[col], errors="coerce")

            conversion_rate = 1 - converted.isna().mean()
            logger.info(f"Column '{col}' datetime conversion success rate: {conversion_rate:.2f}")

            if converted.isna().mean() <= 0.2:
                df[col] = converted
                df = df[(df[col] >= pd.Timestamp("1900-01-01")) & (df[col] <= now)]
                logger.info(f"Column '{col}' converted successfully and filtered for valid ranges.")
            else:
                logger.info(f"Skipping '{col}': low conversion rate.")

    return df


from pandas.api.types import is_datetime64_any_dtype  # add near the top of file
def drop_constant_columns(df):
    """
    Drops columns that have exactly 1 distinct non-NaN value.
    Returns:
        cleaned_df, dropped_columns
    """
    dropped = []

    for col in df.columns:
        # Count unique excluding NaN
        n_unique = df[col].nunique(dropna=True)

        if n_unique <= 1:    # constant column
            dropped.append(col)

    # Drop them
    cleaned_df = df.drop(columns=dropped) if dropped else df

    return cleaned_df, dropped

def detect_numeric_and_categorical(df, target_column=None):
    logger.info("Detecting numeric, categorical, and date/time features...")

    feature_df = df.drop(columns=[target_column], errors="ignore")

    # True numeric columns
    numeric = feature_df.select_dtypes(
        include=["int64", "int32", "float64", "float32"]
    ).columns.tolist()

    # Date / datetime columns – robust detection (handles tz-aware too)
    date_cols = [
        col for col in feature_df.columns
        if is_datetime64_any_dtype(feature_df[col])
    ]

    # Categorical = ONLY object / category / bool (datetimes are excluded)
    categorical = feature_df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    numeric_categorical = []
    numeric_continuous = numeric.copy()

    for col in numeric:
        n_unique = feature_df[col].nunique(dropna=True)
        unique_ratio = n_unique / max(len(feature_df), 1)

        # Only treat *integer-like* very-low-cardinality columns as numeric_categorical
        is_int_like = is_integer_dtype(feature_df[col])

        # MUCH stricter thresholds
        if is_int_like and n_unique <= 15 and unique_ratio <= 0.02:
            numeric_categorical.append(col)
            if col in numeric_continuous:
                numeric_continuous.remove(col)

    logger.info(f"Numeric continuous columns: {numeric_continuous}")
    logger.info(f"Numeric categorical columns: {numeric_categorical}")
    logger.info(f"Categorical columns: {categorical}")
    if date_cols:
        logger.info(
            f"Date/time columns (kept separate, not treated as categorical): {date_cols}"
        )

    return numeric_continuous, numeric_categorical, categorical


def handle_outliers_iqr(df, categorical_columns, target_column=None, multiplier=1.5):
    logger.info("Handling outliers using IQR...")
    before_rows = len(df)

    numeric = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    numeric_continuous = [
        col for col in numeric
        if col not in categorical_columns and col != target_column
    ]

    treat_target_as_categorical = False
    if target_column is not None and target_column in df.columns:
        y = df[target_column]
        is_numeric_target = np.issubdtype(y.dtype, np.number)
        n_unique_y = y.nunique(dropna=True)

        if (not is_numeric_target) or (is_numeric_target and n_unique_y <= 10):
            treat_target_as_categorical = True

    if treat_target_as_categorical:
        logger.info(
            f"Applying IQR outlier removal within categories of target '{target_column}'."
        )
        mask = pd.Series(True, index=df.index)

        for cls_value, group in df.groupby(target_column):
            for col in numeric_continuous:
                Q1 = group[col].quantile(0.25)
                Q3 = group[col].quantile(0.75)
                IQR = Q3 - Q1

                lower = Q1 - multiplier * IQR
                upper = Q3 + multiplier * IQR

                inliers = (group[col] >= lower) & (group[col] <= upper)
                mask.loc[group.index] &= inliers

        df = df[mask]
    else:
        logger.info("Applying global IQR outlier removal.")
        for col in numeric_continuous:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - multiplier * IQR
            upper = Q3 + multiplier * IQR

            logger.info(
                f"IQR bounds for '{col}': {lower:.2f} to {upper:.2f}"
            )
            df = df[(df[col] >= lower) & (df[col] <= upper)]

    removed_rows = before_rows - len(df)
    logger.info(f"Removed {removed_rows} rows due to outliers.")

    return df


def clean_data(df, target_column=None, config=None):
    logger.info("Starting preprocessing pipeline with config...")

    if config is None:
        config = {}

    set_id_index = config.get("set_id_index", True)
    remove_dupes = config.get("remove_duplicates", True)
    handle_dates = config.get("handle_dates", True)
    handle_missing = config.get("handle_missing", True)
    missing_col_threshold = config.get("missing_col_threshold", 0.5)
    missing_row_threshold = config.get("missing_row_threshold", 0.05)
    outlier_strategy = config.get("outlier_strategy", "iqr")
    iqr_multiplier = config.get("iqr_multiplier", 1.5)

    if set_id_index:
        df = set_id_columns_as_index(df)
    id_like = [c for c in df.columns if "id" in c.lower()]
    if id_like:
        df = df.drop(columns=id_like)
        logger.info(f"Dropped ID-like feature columns: {id_like}")
    if remove_dupes:
        df = remove_duplicates(df)

    if handle_dates:
        df = handle_date_columns(df, target_column)

    if handle_missing:
        df = handle_missing_data(
            df,
            col_threshold=missing_col_threshold,
            row_threshold=missing_row_threshold,
            target_column=target_column,
        )

    numeric_continuous, numeric_categorical, categorical = detect_numeric_and_categorical(
        df, target_column
    )

    if outlier_strategy == "iqr":
        df = handle_outliers_iqr(
            df,
            categorical_columns=categorical + numeric_categorical,
            target_column=target_column,
            multiplier=iqr_multiplier,
        )

    return df, numeric_continuous, numeric_categorical, categorical


def save_preprocessed(df, user_id: str):
    from pipeline.utils_paths import get_user_preprocessed

    folder = get_user_preprocessed(user_id)
    path = os.path.join(folder, "cleaned.parquet")
    df.to_parquet(path, index=False)

    logger.info(f"Saved preprocessed data to: {path}")
    return path
