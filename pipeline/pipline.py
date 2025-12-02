from prefect import flow, task, get_run_logger
from prefect.tasks import task_input_hash
from datetime import timedelta
import mysql.connector
import yagmail
import os
import joblib
import numpy as np

# -----------------------------
# ------- DB CONNECTION -------
# -----------------------------
def get_db_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="1234",
        database="student_db"
    )

# Load model and scaler
ARTIFACTS = joblib.load(r"F:\Big_Data\project\NHA-190\models\dropout_model_and_scaler.pkl")
model = ARTIFACTS["model"]
scaler = ARTIFACTS["scaler"]
expected_features = ARTIFACTS["feature_order"]  # safety

# Course inverse mapping
COURSE_INV = {
    12: "Nursing",
    9:  "Management",
    7:  "Informatics Engineering",
    6:  "Veterinary Nursing",
    17: "Management (evening attendance)",
    11: "Tourism",
    13: "Oral Hygiene",
    15: "Journalism and Communication",
    1:  "Biofuel Production Technologies",
    2:  "Animation and Multimedia Design",
    3:  "Social Service (evening attendance)",
    4:  "Agronomy",
    5:  "Communication Design",
    10: "Social Service",
    8:  "Equinculture",
    14: "Advertising and Marketing Management",
    16: "Basic Education"
}

# -----------------------------
# --------- TASKS -------------
# -----------------------------

@task
def predict_and_update_null_targets():
    logger = get_run_logger()
    logger.info("Predicting and updating NULL targets...")

    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    query = """
        SELECT *
        FROM student_data
        WHERE target IS NULL
    """

    cursor.execute(query)
    students = cursor.fetchall()
    logger.info(f"Found {len(students)} students with NULL target")

    for s in students:
        features = (
            s['marital_status'], s['application_mode'], s['application_order'],
            s['course'], s['attendance_regime'], s['previous_qualification'],
            s['nationality'], s['mother_qualification'], s['father_qualification'],
            s['mother_occupation'], s['father_occupation'], s['displaced'],
            s['educational_special_needs'], s['debtor'], s['tuition_fees_up_to_date'],
            s['gender'], s['scholarship_holder'], s['age_at_enrollment'],
            s['international'],
            s['cu_1st_sem_credited'], s['cu_1st_sem_enrolled'], s['cu_1st_sem_evaluations'],
            s['cu_1st_sem_approved'], s['cu_1st_sem_grade'], s['cu_1st_sem_without_evaluation'],
            s['cu_2nd_sem_credited'], s['cu_2nd_sem_enrolled'], s['cu_2nd_sem_evaluations'],
            s['cu_2nd_sem_approved'], s['cu_2nd_sem_grade'], s['cu_2nd_sem_without_evaluation'],
            s['unemployment_rate'], s['inflation_rate'], s['gdp']
        )

        X = np.array(features).reshape(1, -1)
        X_scaled = scaler.transform(X)
        risk_proba = model.predict_proba(X_scaled)[0, 1]
        risk_percent = round(risk_proba * 100, 1)

        update_query = """
            UPDATE student_data
            SET target = %s
            WHERE student_id = %s
        """
        cursor.execute(update_query, (risk_percent, s['student_id']))

    conn.commit()
    cursor.close()
    conn.close()

    logger.info("Updated all NULL targets with predictions")

@task
def fetch_high_target_students():
    logger = get_run_logger()
    logger.info("Connecting to the database...")

    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    query = """
        SELECT student_id, course, nationality, cu_1st_sem_grade, cu_2nd_sem_grade, target
        FROM student_data
        WHERE CAST(target AS DECIMAL(10,2)) > 50
    """

    cursor.execute(query)
    results = cursor.fetchall()
    cursor.close()
    conn.close()

    logger.info(f"Fetched {len(results)} students with target > 50")

    return results

@task
def send_email(students):
    logger = get_run_logger()

    if len(students) == 0:
        logger.info("No students above 50 target. Email will NOT be sent.")
        return

    logger.info("Sending email with student list...")

    # Email Configuration
    sender = "nass147472@gmail.com"
    app_password = "bfei hzmq dphn lese"   # NOT your Gmail password
    receiver = "44mahmoudnasser@gmail.com"

    yag = yagmail.SMTP(sender, app_password)

    # Build email message
    message = "List of students with target > 50:\n\n"
    for s in students:
        course_name = COURSE_INV.get(int(s['course']), f"Code {s['course']}")
        message += f"- ID: {s['student_id']}, Course: {course_name}, Target: {s['target']}\n"

    yag.send(
        to=receiver,
        subject="Students with Target > 50",
        contents=message
    )

    logger.info("Email sent successfully.")

# -----------------------------
# ----------- FLOW ------------
# -----------------------------

@flow(name="Student Target Monitor Pipeline")
def student_pipeline():
    logger = get_run_logger()
    logger.info("Pipeline started.")

    predict_and_update_null_targets()
    students = fetch_high_target_students()
    send_email(students)

    logger.info("Pipeline finished.")

# -----------------------------
# ---- SCHEDULING (5 min) -----
# -----------------------------
if __name__ == "__main__":
    student_pipeline.serve(
        name="Student Target Monitor Service",
        schedule={"interval": timedelta(minutes=5)} # Define the 5-minute schedule
    )