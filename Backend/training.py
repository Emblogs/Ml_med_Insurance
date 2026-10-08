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
            format="%(asctime)s | %(levelname)s | %(message)s",
            handlers=[
                logging.FileHandler(LOG_PATH),
                logging.StreamHandler()
            ]
        )

        #loading data
        df = pd.read_csv(DATASET_PATH)
        logging.info(f"Dataset with shape {df.shape} was loaded")

        #separate x and y
        X = df.drop(columns = [TARGET_COL])
        y = df[TARGET_COL]


        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
        logging.info(f"Dataset has been split into training and test sets with test size {TEST_SIZE}, Train Shape {X_train}, TEST SHAPE {X_test.shape}")

        numerical_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = X_train.select_dtypes(exclude=[np.number]).columns.tolist()




        # numerical features - preprocessing steps
        # imputers can be added in numerical transformer
        numerical_transformer = Pipeline(
        steps=[
        ("scaler", StandardScaler())
        ]
        )

        # categorical features - preprocessing steps
        categorical_transformer = Pipeline(
        steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]
        )

        # preprocessing pipeline
        preprocess = ColumnTransformer(
        transformers=[
        ("num", numerical_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features)
        ]
        )

        #parameters from notebook tuning
        rfc_best = RandomForestRegressor(
            bootstrap=True,
            max_depth=None,
            max_features=0.8,
            min_samples_leaf=4,
            min_samples_split=10,
            n_estimators=200,
            n_jobs=-1,
            random_state=RANDOM_STATE,
            verbose=1
        )

        # Create a pipeline with preprocessing and model
        pipeline = Pipeline([
            ('preprocess', preprocess),
            ('model', rfc_best)
        ])


        #training the model
        pipeline.fit(X_train, y_train)
        
        

        #Evaluation
        y_train_pred = pipeline.predict(X_train)
        y_test_pred = pipeline.predict(X_test)

        rmse_train = root_mean_squared_error(y_train, y_train_pred)
        rmse_test = root_mean_squared_error(y_test, y_test_pred)
        logging.info(f"Model Evaluation:\nRMSE (Train): {rmse_train}\nRMSE (Test): {rmse_test}")

        mae_train = mean_absolute_error(y_train, y_train_pred)
        mae_test = mean_absolute_error(y_test, y_test_pred)
        logging.info(f"Model Evaluation:\nMAE (Train): {mae_train}\nMAE (Test): {mae_test}")

        r2_train = r2_score(y_train, y_train_pred)
        r2_test = r2_score(y_test, y_test_pred)
        logging.info(f"Model Evaluation:\nR2 Score (Train): {r2_train}\nR2 Score (Test): {r2_test}")

        # Save the trained model to a file
        dump(pipeline, MODEL_PATH)

    except Exception as e:
        print(f"Error occured while training model: {e}")

if __name__ == "__main__":   #if this file is run directly by the user
    train_model()

