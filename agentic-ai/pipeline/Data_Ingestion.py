import os
import pandas as pd
import logging
import openpyxl
from pipeline.utils_paths import get_user_uploads

logger = logging.getLogger(__name__)

def load_file(user_id: str, filename: str, url) -> pd.DataFrame:
    try:
        file_path = url
        ext = os.path.splitext(file_path)[-1].lower()

        if ext == ".csv":
            df = pd.read_csv(file_path)

        elif ext in [".xlsx", ".xls"]:
            df = pd.read_excel(file_path, engine="openpyxl")
        elif ext == ".json":
            df = pd.read_json(file_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")

        df = df.loc[:, ~df.columns.str.match(r"^Unnamed")]

        if df.empty:
            raise ValueError(f"Uploaded file '{filename}' is empty.")

        logger.info(
            f"File loaded for user '{user_id}': {filename} ({df.shape[0]} rows × {df.shape[1]} columns)"
        )
        return df

    except Exception as error:
        logger.error(
            f"File ingestion error for user '{user_id}', file '{filename}': {error}",
            exc_info=True,
        )
        raise
