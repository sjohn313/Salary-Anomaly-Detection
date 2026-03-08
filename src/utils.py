import os
import joblib
import logging

def setup_logging():
    """Sets up a basic logger to track progress in the console."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    return logging.getLogger(__name__)

def save_artifact(obj, folder, filename):
    """Saves a python object (like a model or scaler) to a folder."""
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, filename)
    joblib.dump(obj, path)
    print(f"Artifact saved to {path}")

def load_artifact(path):
    """Loads a saved object from a specific path."""
    if os.path.exists(path):
        return joblib.load(path)
    raise FileNotFoundError(f"No file found at {path}")