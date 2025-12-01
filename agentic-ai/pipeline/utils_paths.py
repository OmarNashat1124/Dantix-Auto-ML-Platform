import os

BASE_DIR = "/root/data"


def ensure_folder(path: str):
    os.makedirs(path, exist_ok=True)
    return path


def get_user_folder(user_id: str):
    return ensure_folder(os.path.join(BASE_DIR, "users", user_id))


def get_user_uploads(user_id: str):
    return ensure_folder(os.path.join(BASE_DIR, "users", user_id, "uploads"))


def get_user_preprocessed(user_id: str):
    return ensure_folder(os.path.join(BASE_DIR, "users", user_id, "preprocessed"))


def get_user_engineered(user_id: str):
    return ensure_folder(os.path.join(BASE_DIR, "users", user_id, "engineered"))


def get_user_models(user_id: str):
    return ensure_folder(os.path.join(BASE_DIR, "users", user_id, "models"))


def get_user_visualizations(user_id: str):
    return ensure_folder(os.path.join(BASE_DIR, "users", user_id, "visualizations"))


def get_user_dashboard_json(user_id: str):
    viz_folder = get_user_visualizations(user_id)
    return os.path.join(viz_folder, "dashboard_data.json")

def get_user_reports(user_id: str) -> str:
    return ensure_folder(os.path.join(BASE_DIR, "users", user_id, "reports"))
