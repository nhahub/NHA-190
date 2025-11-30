# api/app.py
from fastapi import FastAPI
import mysql.connector
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Student Dropout Prediction API")

def get_db_connection():
    return mysql.connector.connect(
        host=os.getenv('DB_HOST'),
        user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASSWORD'),
        database=os.getenv('DB_NAME')
    )

@app.get("/predict/{student_id}")
def get_prediction(student_id: int):
    cnx = get_db_connection()
    cursor = cnx.cursor(dictionary=True)
    cursor.execute(
        "SELECT * FROM predictions WHERE student_id = %s ORDER BY prediction_date DESC LIMIT 1",
        (student_id,)
    )
    result = cursor.fetchone()
    cnx.close()
    
    if result:
        return {
            "student_id": result['student_id'],
            "dropout_probability": float(result['dropout_probability']),
            "risk_level": result['risk_level']
        }
    return {"error": "Student not found"}

@app.get("/high_risk_students")
def get_high_risk_students():
    cnx = get_db_connection()
    cursor = cnx.cursor(dictionary=True)
    cursor.execute(
        "SELECT p.student_id, p.dropout_probability, r.target FROM predictions p JOIN raw_students_data r ON p.student_id = r.id WHERE p.risk_level = 'High'"
    )
    results = cursor.fetchall()
    cnx.close()
    return results