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
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mysql.connector
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np

# Load ONCE when server starts
ARTIFACTS = joblib.load(r"F:\Big_Data\project\NHA-190\models\dropout_model_and_scaler.pkl")
model = ARTIFACTS["model"]
scaler = ARTIFACTS["scaler"]
expected_features = ARTIFACTS["feature_order"]  # safety

print("Model and scaler loaded – ready for predictions!")
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],           # Allows your HTML file from anywhere
    allow_credentials=True,
    allow_methods=["*"],           #     Allows POST, OPTIONS, etc.
    allow_headers=["*"],
)
# -------------------------------
#  DICTIONARIES (mapping tables)
# -------------------------------

MARITAL_STATUS = {
    "Single": 1, "Married": 2, "Widower": 3,
    "Divorced": 4, "Facto union": 5, "Legally separated": 6
}

NATIONALITY = {
    "Portuguese": 1, "German": 2, "Spanish": 3, "Italian": 4, "Dutch": 5,
    "English": 6, "Lithuanian": 7, "Angolan": 8, "CapeVerdean": 9,
    "Guinean": 10, "Mozambican": 11, "Santomean": 12, "Turkish": 13,
    "Brazilian": 14, "Romanian": 15, "Moldova": 16, "Mexican": 17,
    "Ukrainian": 18, "Russian": 19, "Cuban": 20, "Colombian": 21
}

APPLICATION_MODE = {
    "1st phase — general contingent": 1,
    "Ordinance No. 612/93": 2,
    "1st phase — special contingent (Azores Island)": 3,
    "Holders of other higher courses": 4,
    "Ordinance No. 854-B/99": 5,
    "International student (bachelor)": 6,
    "1st phase — special contingent (Madeira Island)": 7,
    "2nd phase — general contingent": 8,
    "3rd phase — general contingent": 9,
    "Ordinance No. 533-A/99 item b2": 10,
    "Ordinance No. 533-A/99 item b3": 11,
    "Over 23 years old": 12,
    "Transfer": 13,
    "Change in course": 14,
    "Technological specialization diploma holders": 15,
    "Change in institution/course": 16,
    "Short cycle diploma holders": 17,
    "Change in institution/course (International)": 18
}

COURSE = {
    "Biofuel Production Technologies": 1,
    "Animation and Multimedia Design": 2,
    "Social Service (evening attendance)": 3,
    "Agronomy Course": 4,
    "Communication Design": 5,
    "Veterinary Nursing": 6,
    "Informatics Engineering": 7,
    "Equiniculture": 8,
    "Management": 9,
    "Social Service": 10,
    "Tourism": 11,
    "Nursing": 12,
    "Oral Hygiene": 13,
    "Advertising and Marketing Management": 14,
    "Journalism and Communication": 15,
    "Basic Education": 16,
    "Management (evening attendance)": 17
}

PREVIOUS_QUALIFICATION = {
    "Secondary education": 1,
    "Higher education — bachelor's degree": 2,
    "Higher education — degree": 3,
    "Higher education — master's degree": 4,
    "Higher education — doctorate": 5,
    "Frequency of higher education": 6,
    "12th year — not completed": 7,
    "11th year — not completed": 8,
    "Other — 11th year": 9,
    "10th year": 10,
    "10th year — not completed": 11,
    "Basic education 3rd cycle": 12,
    "Basic education 2nd cycle": 13,
    "Technological specialization course": 14,
    "Degree (1st cycle)": 15,
    "Professional technical course": 16,
    "Master's degree (2nd cycle)": 17,
}

QUALIFICATION = {   # mother & father share same mapping
    "Secondary Education": 1,
    "Bachelor": 2,
    "Degree": 3,
    "Master": 4,
    "Doctorate": 5,
    "Frequency of Higher Education": 6,
    "12th Year — not completed": 7,
    "11th Year — not completed": 8,
    "7th Year (Old)": 9,
    "Other — 11th Year": 10,
    "2nd year complementary": 11,
    "10th Year": 12,
    "General commerce": 13,
    "Basic Education 3rd Cycle": 14,
    "Complementary High School": 15,
    "Technical-professional": 16,
    "Complementary — not concluded": 17,
    "7th year": 18,
    "2nd cycle high school": 19,
    "9th Year — not completed": 20,
    "8th year": 21,
    "Admin & commerce": 22,
    "Accounting": 23,
    "Unknown": 24,
    "Cannot read or write": 25,
    "Can read only": 26,
    "Basic education 1st cycle": 27,
    "Basic education 2nd cycle": 28,
    "Technological specialization": 29,
    "Degree (1st cycle)": 30,
    "Specialized studies": 31,
    "Technical course": 32,
    "Master (2nd cycle)": 33,
    "Doctorate (3rd cycle)": 34
}

OCCUPATION = {
    "Student": 1,
    "Legislative/Executive/Director": 2,
    "Intellectual/Scientific": 3,
    "Technician": 4,
    "Administrative": 5,
    "Services/Security/Sales": 6,
    "Farmers": 7,
    "Construction/Industry": 8,
    "Machine operators": 9,
    "Unskilled workers": 10,
}

GENDER = {"Male": 1, "Female": 0}
ATTENDANCE = {"Daytime": 1, "Evening": 0}
YESNO = {"Yes": 1, "No": 0}
# REVERSE MAPPING — matches your exact COURSE dict above

# -------------------------------
#  INPUT MODEL
# -------------------------------

class Student(BaseModel):
    marital_status: str
    application_mode: str
    application_order: int
    course: str
    attendance_regime: str
    previous_qualification: str
    nationality: str
    mother_qualification: str
    father_qualification: str
    mother_occupation: str
    father_occupation: str
    displaced: str
    educational_special_needs: str
    debtor: str
    tuition_fees_up_to_date: str
    gender: str
    scholarship_holder: str
    age_at_enrollment: int
    international: str
    cu_1st_sem_credited: int
    cu_1st_sem_enrolled: int
    cu_1st_sem_evaluations: int
    cu_1st_sem_approved: int
    cu_1st_sem_grade: float
    cu_1st_sem_without_evaluation: int
    cu_2nd_sem_credited: int
    cu_2nd_sem_enrolled: int
    cu_2nd_sem_evaluations: int
    cu_2nd_sem_approved: int
    cu_2nd_sem_grade: float
    cu_2nd_sem_without_evaluation: int
    unemployment_rate: float
    inflation_rate: float
    gdp: float


# -------------------------------
#  DATABASE CONNECTION
# -------------------------------
def get_db():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="1234",
        database="student_db"
    )


# -------------------------------
#  CONVERT FUNCTION
# -------------------------------
def map_value(value, mapping, field):
    if value not in mapping:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid value '{value}' for field '{field}'"
        )
    return mapping[value]


# -------------------------------
#  API ROUTE
# -------------------------------
@app.post("/add_student")
def add_student(data: Student):
    # --- 1. Build numeric feature row (same as before) ---
    row = (
        map_value(data.marital_status, MARITAL_STATUS, "marital_status"),
        map_value(data.application_mode, APPLICATION_MODE, "application_mode"),
        data.application_order,
        map_value(data.course, COURSE, "course"),
        map_value(data.attendance_regime, ATTENDANCE, "attendance_regime"),
        map_value(data.previous_qualification, PREVIOUS_QUALIFICATION, "previous_qualification"),
        map_value(data.nationality, NATIONALITY, "nationality"),
        map_value(data.mother_qualification, QUALIFICATION, "mother_qualification"),
        map_value(data.father_qualification, QUALIFICATION, "father_qualification"),
        map_value(data.mother_occupation, OCCUPATION, "mother_occupation"),
        map_value(data.father_occupation, OCCUPATION, "father_occupation"),
        map_value(data.displaced, YESNO, "displaced"),
        map_value(data.educational_special_needs, YESNO, "educational_special_needs"),
        map_value(data.debtor, YESNO, "debtor"),
        map_value(data.tuition_fees_up_to_date, YESNO, "tuition_fees_up_to_date"),
        map_value(data.gender, GENDER, "gender"),
        map_value(data.scholarship_holder, YESNO, "scholarship_holder"),
        data.age_at_enrollment,
        map_value(data.international, YESNO, "international"),
        data.cu_1st_sem_credited,
        data.cu_1st_sem_enrolled,
        data.cu_1st_sem_evaluations,
        data.cu_1st_sem_approved,
        data.cu_1st_sem_grade,
        data.cu_1st_sem_without_evaluation,
        data.cu_2nd_sem_credited,
        data.cu_2nd_sem_enrolled,
        data.cu_2nd_sem_evaluations,
        data.cu_2nd_sem_approved,
        data.cu_2nd_sem_grade,
        data.cu_2nd_sem_without_evaluation,
        data.unemployment_rate,
        data.inflation_rate,
        data.gdp
    )

 
    # --- 3. Insert into DB with predicted target ---
    db = None
    cursor = None
    try:
        db = get_db()
        cursor = db.cursor()

        sql = """
            INSERT INTO student_data (
                marital_status, application_mode, application_order, course,
                attendance_regime, previous_qualification, nationality,
                mother_qualification, father_qualification, mother_occupation,
                father_occupation, displaced, educational_special_needs, debtor,
                tuition_fees_up_to_date, gender, scholarship_holder,
                age_at_enrollment, international, cu_1st_sem_credited,
                cu_1st_sem_enrolled, cu_1st_sem_evaluations, cu_1st_sem_approved,
                cu_1st_sem_grade, cu_1st_sem_without_evaluation,
                cu_2nd_sem_credited, cu_2nd_sem_enrolled, cu_2nd_sem_evaluations,
                cu_2nd_sem_approved, cu_2nd_sem_grade,
                cu_2nd_sem_without_evaluation, unemployment_rate,
                inflation_rate, gdp, target
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,
                      %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s, %s)
        """

        # Add the predicted target at the end
        row_with_target = row + (None,)  # None في بايثون هيتحول لـ NULL في قاعدة البيانات
        cursor.execute(sql, row_with_target)
        db.commit()
        inserted_id = cursor.lastrowid

    except Exception as e:
        if db:
            db.rollback()
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

    finally:
        if cursor:
            cursor.close()
        if db:
            db.close()

 

    return {
        "status": "success",
        "inserted_id": inserted_id,
        "message": f"Student added!"
    }
@app.options("/add_student")  # <-- ADD THIS
def options_add_student():
    return {}
# Add this reverse mapping at the top (after your COURSE dict)
COURSE_INV = {v: k for k, v in COURSE.items()}
GENDER_INV = {0: "Female", 1: "Male"}
YESNO_INV = {0: "No", 1: "Yes"}
@app.get("/high_risk_students")
def get_high_risk_students():
    db = None
    cursor = None
    try:
        db = get_db()
        cursor = db.cursor(dictionary=True)

        cursor.execute("""
            SELECT 
                student_id,
                marital_status, application_mode, application_order, course,
                attendance_regime, previous_qualification, nationality,
                mother_qualification, father_qualification, mother_occupation,
                father_occupation, displaced, educational_special_needs, debtor,
                tuition_fees_up_to_date, gender, scholarship_holder,
                age_at_enrollment, international,
                cu_1st_sem_credited, cu_1st_sem_enrolled, cu_1st_sem_evaluations,
                cu_1st_sem_approved, cu_1st_sem_grade, cu_1st_sem_without_evaluation,
                cu_2nd_sem_credited, cu_2nd_sem_enrolled, cu_2nd_sem_evaluations,
                cu_2nd_sem_approved, cu_2nd_sem_grade, cu_2nd_sem_without_evaluation,
                unemployment_rate, inflation_rate, gdp,
                target
            FROM student_data
            ORDER BY student_id DESC
        """)
        all_students = cursor.fetchall()

        high_risk_list = []

        for s in all_students:
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

            if risk_percent > 50:
                high_risk_list.append({
                    "id": s['student_id'],  
                    "course": COURSE_INV.get(int(s['course']), f"Code {s['course']}"),
                    "age_at_enrollment": s['age_at_enrollment'],
                    "gender": "Male" if s['gender'] == 1 else "Female",   # Fixed: was string before
                    "debtor": "Yes" if s['debtor'] == 1 else "No",       # Fixed: was string before
                    "scholarship_holder": "Yes" if s['scholarship_holder'] == 1 else "No",
                    "dropout_risk_percent": risk_percent,
                    "predicted_class": int(model.predict(X_scaled)[0])
                })

        high_risk_list.sort(key=lambda x: x["dropout_risk_percent"], reverse=True)
        return high_risk_list

    except Exception as e:
        print("Error in high_risk_students:", e)
        raise HTTPException(status_code=500, detail="Failed to load high-risk students")
    finally:
        if cursor:
            cursor.close()
        if db:
            db.close()
@app.post("/predict_live")
def predict_live(data: dict):
    try:
        # Your existing mapping + scaling code here
        features = map_and_scale(data)  # your function
        risk = model.predict_proba(features)[0][1] * 100
        return {"risk": round(risk, 1)}
    except:
        return {"risk": 0}