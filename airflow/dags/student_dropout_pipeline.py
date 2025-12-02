from prefect import flow, task
from datetime import datetime
import joblib
import pandas as pd
import logging
import os

# ---------------------------- Logging ---------------------------- #
LOG_FILE = "pipeline_logs.txt"
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ---------------------------- Tasks ----------------------------- #

@task
def load_data():
    logging.info("Loading input data...")
    df = pd.read_csv("input_data.csv")
    logging.info(f"Loaded data shape: {df.shape}")
    return df

@task
def preprocess(df):
    logging.info("Preprocessing data...")
    df = df.fillna(0)
    logging.info("Preprocessing completed")
    return df

@task
def load_model():
    logging.info("Loading model & scaler...")
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    logging.info("Model and scaler loaded")
    return model, scaler

@task
def predict(df, model, scaler):
    logging.info("Running prediction...")
    X = scaler.transform(df)
    preds = model.predict(X)
    logging.info(f"Predictions completed: {len(preds)} records")
    return preds

@task
def save_output(predictions):
    logging.info("Saving output...")
    output_file = f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    pd.DataFrame({"prediction": predictions}).to_csv(output_file, index=False)
    logging.info(f"Output saved: {output_file}")
    return output_file

# ---------------------------- Flow ------------------------------ #

@flow(name="Student Dropout Prediction Pipeline")
def student_pipeline():
    logging.info("=== PIPELINE STARTED ===")

    df = load_data()
    df_processed = preprocess(df)
    model, scaler = load_model()
    predictions = predict(df_processed, model, scaler)
    save_output(predictions)

    logging.info("=== PIPELINE FINISHED ===")

# ---------------------------- Trigger ---------------------------- #

if __name__ == "__main__":
    student_pipeline()
