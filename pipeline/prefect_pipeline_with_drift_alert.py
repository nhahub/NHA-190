# drift_monitor_pipeline.py

from prefect import flow, task, get_run_logger
from datetime import timedelta
import requests
import yagmail
import pandas as pd

# -----------------------------
# --------- CONFIG ------------
# -----------------------------
DRIFT_API_URL = "http://127.0.0.1:8000/data_drift"   # Your FastAPI drift endpoint
EMAIL_SENDER = "nass147472@gmail.com"
EMAIL_APP_PASSWORD = "bfei hzmq dphn lese"
EMAIL_RECEIVER = "44mahmoudnasser@gmail.com"

# -----------------------------
# --------- TASKS -------------
# -----------------------------

@task
def check_drift():
    logger = get_run_logger()
    logger.info("Calling /data_drift endpoint...")

    try:
        response = requests.get(DRIFT_API_URL, timeout=30)
        response.raise_for_status()
        report = response.json()

        logger.info(f"Drift report received | dataset_drift_detected = {report['dataset_drift_detected']}")
        logger.info(f"Drift share: {report['drift_share_percent']}% | Recommendation: {report['recommendation']}")

        return report

    except Exception as e:
        logger.error(f"Failed to reach drift endpoint: {e}")
        return {"dataset_drift_detected": False, "error": str(e)}


@task
def send_drift_alert_email(drift_report: dict):
    logger = get_run_logger()

    if not drift_report.get("dataset_drift_detected", False):
        logger.info("No significant drift detected → No email sent.")
        return

    logger.warning("DATA DRIFT DETECTED → Sending alert email!")

    yag = yagmail.SMTP(EMAIL_SENDER, EMAIL_APP_PASSWORD)

    # Build rich HTML email
    drifted_features = "<br>".join([f"• {f}" for f in drift_report.get("drifted_features", [])[:15]])
    if not drifted_features:
        drifted_features = "• Not specified in report"

    subject = "DATA DRIFT DETECTED – Retrain Model ASAP!"

    body = f"""
    <h2 style="color:#e74c3c;">Data Drift Detected in Student Dropout Model</h2>
    <p><strong>Detected at:</strong> {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    <p><strong>Drift Level:</strong> <span style="font-size:18px;color:#e67e22;font-weight:bold;">{drift_report['drift_share_percent']}%</span> of features drifted</p>
    <p><strong>Drifted Features Count:</strong> {drift_report['drifted_features_count']}</p>
    <p><strong>Top Drifted Features:</strong></p>
    <ul style="font-size:15px;">
        {drifted_features}
    </ul>
    <hr>
    <p><strong>Recommendation:</strong> <span style="color:#c0392b;font-weight:bold;">{drift_report['recommendation']}</span></p>
    <p><em>This is an automated alert from your Data Drift Monitoring Pipeline.</em></p>
    """

    yag.send(
        to=EMAIL_RECEIVER,
        subject=subject,
        contents=body
    )

    logger.info("Drift alert email sent successfully!")


# -----------------------------
# ----------- FLOW ------------
# -----------------------------

@flow(name="Data Drift Monitoring Pipeline")
def drift_monitor_pipeline():
    logger = get_run_logger()
    logger.info("=== Data Drift Monitoring Pipeline Started ===")

    report = check_drift()
    send_drift_alert_email(report)

    logger.info("=== Data Drift Monitoring Pipeline Finished ===")


# -----------------------------
# ---- SCHEDULING (every 30 min) -----
# -----------------------------
if __name__ == "__main__":
    drift_monitor_pipeline.serve(
        name="Data Drift Monitor Service",
        schedule={"interval": timedelta(minutes=30)} # Define the 5-minute schedule
    
    )