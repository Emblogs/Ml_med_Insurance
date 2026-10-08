import os
import logging
from dotenv import load_dotenv
from joblib import dump
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import (
    root_mean_squared_error,
    mean_absolute_error,
    r2_score
)

def train_model():
    try:
        #load our environment variables
        load_dotenv()

        PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT")).resolve()

        DATASET_PATH = PROJECT_ROOT / os.getenv("DATASET_DIR") / os.getenv("DATASET_NAME")
        MODEL_PATH = PROJECT_ROOT / os.getenv("MODEL_DIR") / os.getenv("MODEL_NAME")
        LOG_PATH = PROJECT_ROOT / os.getenv("LOG_DIR") / os.getenv("LOG_NAME")

        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

        TARGET_COL = os.getenv("TARGET_COL")
        RANDOM_STATE = int(os.getenv("RANDOM_STATE"))
        TEST_SIZE = float(os.getenv("TEST_SIZE"))

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
        )   ha
    except Exception as e:
        print(f"Error in training the model: {e}")

if __name__ == "__main__":
    train_model()