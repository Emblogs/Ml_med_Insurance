import os
import logging
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from joblib import dump, load

#load env content to env vars
load_dotenv()

PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT")).resolve()
MODEL_PATH = PROJECT_ROOT / os.getenv("MODEL_DIR") / os.getenv("MODEL_NAME")
LOG_PATH = PROJECT_ROOT / os.getenv("LOG_DIR") /os.getenv("LOG_NAME")

LOG_PATH.parent.mkdir(parents= True, exist_ok = True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_PATH)
    ]

)

#load the trained model--only once
model = load(MODEL_PATH)
logging.info("Model has been loaded successfully")

def predict_insurance_charges(input_data:dict):
    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data])
    logging.info(f"Input data converted to DataFrame with shape {input_df.shape}")

    # Make prediction
    prediction = model.predict(input_df)[0]
    logging.info(f"Prediction made successfully: {prediction}")

    return prediction
 