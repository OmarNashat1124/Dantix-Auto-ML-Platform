import os
import json
import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional

from pipeline.utils_paths import get_user_visualizations, get_user_dashboard_json

logger = logging.getLogger(__name__)

# ============================================================
# ALLOWED CHART TYPES
# ============================================================

ALLOWED_CHART_TYPES = [
    "histogram",    # Distribution of numeric variable
    "bar",          # Categorical comparison
    "pie",          # Proportion analysis (small categories)
    "box",          # Box plot with outliers
    "violin",       # Distribution shape
    "scatter",      # Two numeric variables correlation
    "heatmap",      # Correlation matrix
    "line",         # Time series / sequential data
]


# ============================================================
# JSON SAFETY
# ============================================================

def json_safe(obj: Any) -> Any:
    """Convert pandas/numpy objects to JSON-safe values."""
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_safe(v) for v in obj]
    if isinstance(obj, (pd.Timestamp, np.datetime64)):
        return str(obj)
    if isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    if isinstance(obj, (np.float64, np.float32, np.float16)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return json_safe(obj.tolist())
    if pd.isna(obj):
        return None
    return obj


# ============================================================
# CHART DATA GENERATORS
# ============================================================

def generate_histogram_data(df: pd.DataFrame, col: str, bins: int = 20) -> Dict[str, Any]:
    """Generate histogram data ready for plotting."""
    data = df[col].dropna()
    if len(data) == 0:
        return {"x": [], "y": [], "bin_edges": []}
    
    hist, bin_edges = np.histogram(data, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    return {
        "x": bin_centers.tolist(),
        "y": hist.tolist(),
        "bin_edges": bin_edges.tolist(),
        "stats": {
            "mean": float(data.mean()),
            "median": float(data.median()),
            "std": float(data.std()),
            "min": float(data.min()),
            "max": float(data.max()),
        }
    }


def generate_bar_data(df: pd.DataFrame, col: str, y_col: Optional[str] = None) -> Dict[str, Any]:
    """Generate bar chart data ready for plotting."""
    if y_col:
        grouped = df.groupby(col)[y_col].agg(['mean', 'count', 'std']).reset_index()
        grouped = grouped.sort_values('mean', ascending=False).head(20)
        
        return {
            "labels": grouped[col].tolist(),
            "values": grouped['mean'].round(2).tolist(),
            "counts": grouped['count'].tolist(),
            "std": grouped['std'].fillna(0).round(2).tolist(),
            "type": "aggregated",
            "y_label": f"Mean {y_col}",
        }
    else:
        value_counts = df[col].value_counts().head(20)
        total = value_counts.sum()
        
        return {
            "labels": value_counts.index.tolist(),
            "values": value_counts.tolist(),
            "percentages": (value_counts / total * 100).round(2).tolist(),
            "type": "count",
            "y_label": "Count",
        }


def generate_scatter_data(df: pd.DataFrame, x_col: str, y_col: str, sample_size: int = 1000) -> Dict[str, Any]:
    """Generate scatter plot data ready for plotting."""
    plot_df = df[[x_col, y_col]].dropna()
    
    if len(plot_df) > sample_size:
        plot_df = plot_df.sample(sample_size, random_state=42)
    
    correlation = df[[x_col, y_col]].corr().iloc[0, 1]
    
    return {
        "x": plot_df[x_col].tolist(),
        "y": plot_df[y_col].tolist(),
        "correlation": float(correlation) if pd.notna(correlation) else None,
        "n_points": len(plot_df),
    }


def generate_box_data(df: pd.DataFrame, y_col: str, x_col: Optional[str] = None) -> Dict[str, Any]:
    """Generate box plot data ready for plotting."""
    if x_col:
        categories = df[x_col].unique()[:20]
        box_data = []
        
        for cat in categories:
            subset = df[df[x_col] == cat][y_col].dropna()
            if len(subset) > 0:
                q1 = subset.quantile(0.25)
                q2 = subset.quantile(0.50)
                q3 = subset.quantile(0.75)
                iqr = q3 - q1
                lower_fence = q1 - 1.5 * iqr
                upper_fence = q3 + 1.5 * iqr
                
                outliers = subset[(subset < lower_fence) | (subset > upper_fence)]
                
                box_data.append({
                    "category": str(cat),
                    "min": float(subset.min()),
                    "q1": float(q1),
                    "median": float(q2),
                    "q3": float(q3),
                    "max": float(subset.max()),
                    "outliers": outliers.tolist()[:50],
                    "count": len(subset),
                })
        
        return {
            "data": box_data,
            "type": "grouped",
        }
    else:
        data = df[y_col].dropna()
        if len(data) == 0:
            return {"data": [], "type": "single"}
        
        q1 = data.quantile(0.25)
        q2 = data.quantile(0.50)
        q3 = data.quantile(0.75)
        iqr = q3 - q1
        lower_fence = q1 - 1.5 * iqr
        upper_fence = q3 + 1.5 * iqr
        
        outliers = data[(data < lower_fence) | (data > upper_fence)]
        
        return {
            "data": [{
                "category": y_col,
                "min": float(data.min()),
                "q1": float(q1),
                "median": float(q2),
                "q3": float(q3),
                "max": float(data.max()),
                "outliers": outliers.tolist()[:50],
                "count": len(data),
            }],
            "type": "single",
        }


def generate_pie_data(df: pd.DataFrame, col: str, max_categories: int = 8) -> Dict[str, Any]:
    """Generate pie chart data ready for plotting."""
    value_counts = df[col].value_counts().head(max_categories)
    total = value_counts.sum()
    
    if len(df[col].unique()) > max_categories:
        other_count = len(df) - total
        if other_count > 0:
            value_counts['Other'] = other_count
            total = len(df)
    
    return {
        "labels": value_counts.index.tolist(),
        "values": value_counts.tolist(),
        "percentages": (value_counts / total * 100).round(2).tolist(),
    }


def generate_heatmap_data(df: pd.DataFrame, numeric_cols: List[str]) -> Dict[str, Any]:
    """Generate correlation heatmap data ready for plotting."""
    cols = [c for c in numeric_cols if c in df.columns]
    if len(cols) < 2:
        return {"z": [], "x": [], "y": []}
    
    corr_matrix = df[cols].corr().fillna(0.0)
    
    return {
        "z": corr_matrix.values.tolist(),
        "x": corr_matrix.columns.tolist(),
        "y": corr_matrix.index.tolist(),
        "colorscale": "RdBu",
        "zmin": -1,
        "zmax": 1,
    }


# ============================================================
# STATISTICS
# ============================================================

def compute_summary_statistics(
    df: pd.DataFrame,
    numeric_continuous: List[str],
    numeric_categorical: List[str],
    categorical_columns: List[str],
) -> Dict[str, Any]:
    """Compute summary statistics for dashboard."""
    summary = {"continuous": {}, "numeric_categorical": {}, "categorical": {}}

    for col in numeric_continuous:
        if col not in df.columns:
            continue
        s = df[col].dropna()
        if len(s) == 0:
            continue
        summary["continuous"][col] = {
            "count": int(s.count()),
            "mean": float(s.mean()),
            "std": float(s.std()),
            "min": float(s.min()),
            "25%": float(s.quantile(0.25)),
            "50%": float(s.quantile(0.50)),
            "75%": float(s.quantile(0.75)),
            "max": float(s.max()),
        }

    for col in numeric_categorical:
        if col not in df.columns:
            continue
        summary["numeric_categorical"][col] = {
            "counts": df[col].value_counts().to_dict(),
            "unique": int(df[col].nunique()),
        }

    for col in categorical_columns:
        if col not in df.columns:
            continue
        vc = df[col].value_counts()
        summary["categorical"][col] = {
            "counts": vc.head(20).to_dict(),
            "unique": int(df[col].nunique()),
        }

    return summary


def compute_correlation_matrix(
    df: pd.DataFrame,
    numeric_continuous: List[str]
) -> Dict[str, Any]:
    """Compute correlation matrix."""
    cols = [c for c in numeric_continuous if c in df.columns]
    if len(cols) < 2:
        return {}
    
    corr = df[cols].corr(method='pearson')
    
    top_corr = []
    for i, c1 in enumerate(cols):
        for c2 in cols[i+1:]:
            val = corr.loc[c1, c2]
            if pd.notna(val):
                top_corr.append({
                    "feature_1": c1,
                    "feature_2": c2,
                    "correlation": float(val),
                })
    
    top_corr.sort(key=lambda x: abs(x["correlation"]), reverse=True)
    
    return {
        "matrix": corr.replace({np.nan: None}).to_dict(),
        "top_correlations": top_corr[:15],
    }


def compute_categorical_distributions(
    df: pd.DataFrame,
    categorical_columns: List[str],
    numeric_categorical: List[str],
) -> Dict[str, Any]:
    """Compute distributions for categorical columns."""
    dist = {}
    all_cats = list(set(categorical_columns + numeric_categorical))
    
    for col in all_cats:
        if col not in df.columns:
            continue
        vc = df[col].value_counts()
        dist[col] = {
            "counts": vc.head(20).to_dict(),
            "frequency": (vc / len(df)).head(20).to_dict(),
        }
    
    return dist


def build_ready_charts_from_plan(
    df: pd.DataFrame,
    viz_plan: Dict[str, Any],
    numeric_continuous: List[str],
    numeric_categorical: List[str],
    categorical_columns: List[str],
    target_column: Optional[str] = None,
    max_charts: int = 20,
) -> List[Dict[str, Any]]:
    """
    Take the LLM viz_plan (metadata only) and attach READY-TO-PLOT data
    using the same structure as the old static charts.
    """
    charts_out: List[Dict[str, Any]] = []
    seen_ids = set()

    plan_charts = viz_plan.get("charts") or []
    for idx, spec in enumerate(plan_charts[:max_charts], start=1):
        ctype = (spec.get("type") or "").lower()
        if ctype not in ALLOWED_CHART_TYPES:
            continue

        x = spec.get("x")
        y = spec.get("y")
        desc = spec.get("description") or spec.get("title") or ""
        width = spec.get("width") or 6
        order = spec.get("order") or idx

        cid = spec.get("id") or f"{ctype}_{idx}"
        if cid in seen_ids:
            cid = f"{cid}_{idx}"
        seen_ids.add(cid)

        chart = {
            "id": cid,
            "type": ctype,
            "x": x,
            "y": y,
            "title": desc,
            "width": width,
            "order": order,
        }

        try:
            if ctype == "histogram":
                if not x or x not in df.columns:
                    continue
                chart["data"] = generate_histogram_data(df, x)

            elif ctype == "scatter":
                if not x or not y or x not in df.columns or y not in df.columns:
                    continue
                chart["data"] = generate_scatter_data(df, x, y)

            elif ctype == "bar":
                if not x or x not in df.columns:
                    continue
                agg_col = y if (y and y in df.columns) else None
                chart["data"] = generate_bar_data(df, x, agg_col)

            elif ctype == "box":
                # y MUST be numeric; x optional categorical
                if not y or y not in df.columns:
                    continue
                group_col = x if (x and x in df.columns) else None
                chart["data"] = generate_box_data(df, y, group_col)

            elif ctype == "pie":
                if not x or x not in df.columns:
                    continue
                chart["data"] = generate_pie_data(df, x)

            elif ctype == "heatmap":
                # ignore x/y, use numeric_continuous intersection
                chart["data"] = generate_heatmap_data(df, numeric_continuous)

            elif ctype == "line":
                # simple version: same data layout as scatter
                if not x or not y or x not in df.columns or y not in df.columns:
                    continue
                chart["data"] = generate_scatter_data(df, x, y)

            else:
                # unsupported type → skip
                continue

        except Exception as e:
            logger.warning(f"Failed to build chart data for LLM chart {cid}: {e!r}")
            continue

        charts_out.append(chart)

    return charts_out


def generate_dashboard_data(
    df: pd.DataFrame,
    user_id: str,
    numeric_continuous: List[str],
    numeric_categorical: List[str],
    categorical_columns: List[str],
    date_columns: Optional[List[str]] = None, 
    target_column: Optional[str] = None,
    df_raw_for_counts: Optional[pd.DataFrame] = None,
) -> str:
    """
    Generate dashboard JSON data WITH READY-TO-PLOT DATA.
    
    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataframe (already preprocessed)
    user_id : str
        User identifier
    numeric_continuous, numeric_categorical, categorical_columns : List[str]
        Column classifications from preprocessing
    target_column : str, optional
        Target column name
        
    Returns
    -------
    str
        Path to saved dashboard JSON
    """
    folder = get_user_visualizations(user_id)
    os.makedirs(folder, exist_ok=True)
    dashboard_file = get_user_dashboard_json(user_id)

    df_for_counts = df_raw_for_counts if df_raw_for_counts is not None else df
    dashboard_data = {
        "user_id": user_id,
        "columns": df.columns.tolist(),
        "target_column": target_column,
        "numeric_continuous": numeric_continuous,
        "numeric_categorical": numeric_categorical,
        "categorical_columns": categorical_columns,
        "date_columns": date_columns or [],
        "charts": []
    }

    safe_data = json_safe(dashboard_data)

    with open(dashboard_file, "w") as f:
        json.dump(safe_data, f, indent=2)

    logger.info(f"Dashboard saved for user {user_id}: {dashboard_file}")
    return dashboard_file
