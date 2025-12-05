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
import logging
from datetime import datetime
from fastapi import Request
import json

# Create a simple file logger
logging.basicConfig(
    filename="api_usage.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

logger = logging.getLogger("api_logger")
# ... rest of your imports
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

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = datetime.now()
    client_ip = request.client.host
    method = request.method
    path = request.url.path
    
    try:
        response = await call_next(request)
        status_code = response.status_code
    except Exception as e:
        status_code = 500
        response = None

    duration = (datetime.now() - start_time).total_seconds() * 1000  # ms

    log_entry = {
        "timestamp": start_time.isoformat(),
        "ip": client_ip,
        "method": method,
        "endpoint": path,
        "status": status_code,
        "duration_ms": round(duration, 2)
    }

    logger.info(json.dumps(log_entry))
    return response 
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

        # Simple, fast, 100% consistent with Prefect pipeline
        query = """
            SELECT 
                student_id,
                course,
                age_at_enrollment,
                gender,
                debtor,
                scholarship_holder,
                target
            FROM student_data
            WHERE target IS NOT NULL 
              AND target > 25
            ORDER BY target DESC
        """

        cursor.execute(query)
        rows = cursor.fetchall()

        result = []
        for row in rows:
            result.append({
                "id": row["student_id"],
                "course": COURSE_INV.get(int(row["course"]), "Unknown Course"),
                "age_at_enrollment": row["age_at_enrollment"],
                "gender": "Male" if row["gender"] == 1 else "Female",
                "debtor": "Yes" if row["debtor"] == 1 else "No",
                "scholarship_holder": "Yes" if row["scholarship_holder"] == 1 else "No",
                "dropout_risk_percent": round(float(row["target"]), 1),   # already saved by Prefect
            })

        return result

    except Exception as e:
        print("Error:", e)
        raise HTTPException(status_code=500, detail="Failed to load high-risk students")
    finally:
        if cursor: cursor.close()
        if db: db.close()

@app.post("/predict_live")
def predict_live(data: dict):
    try:
        # Your existing mapping + scaling code here
        features = map_and_scale(data)  # your function
        risk = model.predict_proba(features)[0][1] * 100
        return {"risk": round(risk, 1)}
    except:
        return {"risk": 0}\
        

from fastapi import FastAPI, HTTPException
from typing import Dict, Any
import pandas as pd

from fastapi import FastAPI, HTTPException
from typing import Dict, Any, List
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp, chi2_contingency
import warnings
warnings.filterwarnings("ignore")

def safe_chi2_test(series_ref: pd.Series, series_curr: pd.Series, min_obs: int = 10) -> float:
    """
    Safe chi-square test that never crashes.
    Returns p-value (1.0 = no evidence of drift)
    """
    if len(series_ref) == 0 or len(series_curr) == 0:
        return 1.0

    # Combine and get all possible categories
    combined = pd.concat([series_ref, series_curr], ignore_index=True)
    if combined.nunique() < 2:
        return 1.0  # only one category → no drift possible

    # Create contingency table: rows = categories, columns = [ref, curr]
    ref_counts = series_ref.value_counts()
    curr_counts = series_curr.value_counts()
    all_categories = ref_counts.index.union(curr_counts.index)

    ref_counts = ref_counts.reindex(all_categories, fill_value=0)
    curr_counts = curr_counts.reindex(all_categories, fill_value=0)

    contingency = np.array([ref_counts.values, curr_counts.values])

    # Safety checks
    if contingency.size == 0:
        return 1.0
    if np.any(contingency < 0):
        return 1.0
    # If total observations too low → skip chi2 (avoid unreliable p-values)
    if contingency.sum() < min_obs:
        return 1.0
    # If any expected frequency would be < 5 → use higher p-value (conservative)
    row_sums = contingency.sum(axis=1)
    col_sums = contingency.sum(axis=0)
    total = contingency.sum()
    expected = np.outer(row_sums, col_sums) / total
    if np.any(expected < 5):
        # Fall back to Fisher's exact if 2x2, otherwise be conservative
        if contingency.shape == (2, 2):
            from scipy.stats import fisher_exact
            oddsratio, p = fisher_exact(contingency)
            return p
        else:
            return 0.05  # borderline: flag as possible drift but don't crash

    chi2, p, dof, _ = chi2_contingency(contingency, correction=True)
    return p

from fastapi.responses import JSONResponse
import pandas as pd
from scipy.stats import ks_2samp

@app.get("/data_drift")
def get_data_drift_report():
    db = get_db()
    cursor = db.cursor(dictionary=True)
    try:
        cursor.execute("SELECT * FROM student_training_data")
        ref_rows = cursor.fetchall()
        cursor.execute("SELECT * FROM student_data WHERE target IS NOT NULL")
        curr_rows = cursor.fetchall()

        ref_df = pd.DataFrame(ref_rows)
        curr_df = pd.DataFrame(curr_rows)

        if len(curr_df) < 20:
            return JSONResponse(content={
                "dataset_drift_detected": False,
                "drift_share_percent": 0.0,
                "drifted_features_count": 0,
                "drifted_features": [],
                "target_drift_p_value": 1.0000,
                "recommendation": "Not enough data",
                "reference_size": len(ref_df),
                "current_size": len(curr_df)
            })

        # Clean
        for df in [ref_df, curr_df]:
            df.drop(columns=['student_id'], errors='ignore', inplace=True)

        common_cols = ref_df.columns.intersection(curr_df.columns)
        ref_df = ref_df[common_cols].copy()
        curr_df = curr_df[common_cols].copy()

        categorical = ['marital_status', 'application_mode', 'course', 'attendance_regime',
                       'previous_qualification', 'nationality', 'mother_qualification',
                       'father_qualification', 'mother_occupation', 'father_occupation',
                       'displaced', 'educational_special_needs', 'debtor',
                       'tuition_fees_up_to_date', 'gender', 'scholarship_holder', 'international']
        numerical = [c for c in common_cols if c not in categorical and c != 'target']

        drifted = []

        # Categorical drift
        for col in [c for c in categorical if c in common_cols]:
            r_freq = ref_df[col].value_counts(normalize=True)
            c_freq = curr_df[col].value_counts(normalize=True)
            all_cats = r_freq.index.union(c_freq.index)
            r_freq = r_freq.reindex(all_cats, fill_value=0)
            c_freq = c_freq.reindex(all_cats, fill_value=0)
            freq_diff = (r_freq - c_freq).abs().sum()
            if freq_diff > 0.3:
                drifted.append(f"{col} (cat, freq_diff={freq_diff:.2f})")

        # Numerical drift
        for col in numerical:
            try:
                r = pd.to_numeric(ref_df[col], errors='coerce').dropna()
                c = pd.to_numeric(curr_df[col], errors='coerce').dropna()
                if len(r) < 10 or len(c) < 10: continue
                p = ks_2samp(r, c).pvalue
                shift = abs(r.mean() - c.mean()) / ((r.std() + c.std()) / 2) if (r.std() + c.std()) > 0 else 0
                if p < 0.05 or shift > 0.5:
                    drifted.append(f"{col} (num, p={p:.3f}, shift={shift:.1f}σ)")
            except:
                pass

        # Target drift p-value
        target_p = 1.0
        if 'target' in ref_df.columns and 'target' in curr_df.columns:
            try:
                r_t = pd.to_numeric(ref_df['target'], errors='coerce').dropna()
                c_t = pd.to_numeric(curr_df['target'], errors='coerce').dropna()
                if len(r_t) > 10 and len(c_t) > 10:
                    target_p = ks_2samp(r_t, c_t).pvalue
            except:
                pass

        total_features = len(numerical) + len([c for c in categorical if c in common_cols])
        drift_ratio = len(drifted) / max(total_features, 1)
        drift_detected = len(drifted) >= 2 or drift_ratio > 0.15 or target_p < 0.05

        result = {
            "dataset_drift_detected": bool(drift_detected),
            "drift_share_percent": round(drift_ratio * 100, 2),
            "drifted_features_count": len(drifted),
            "drifted_features": drifted[:20],
            "target_drift_p_value": round(target_p, 6),
            "recommendation": "DRIFT DETECTED — RETRAIN MODEL!" if drift_detected else "Model is healthy",
            "reference_size": len(ref_df),
            "current_size": len(curr_df),
            "generated_at": pd.Timestamp.now().isoformat(),
        }

        return JSONResponse(content=result)

    except Exception as e:
        return JSONResponse(content={
            "dataset_drift_detected": False,
            "drift_share_percent": 0,
            "drifted_features": [],
            "target_drift_p_value": 1.0000,
            "recommendation": "Error occurred",
            "error": str(e)
        }, status_code=500)
    finally:
        cursor.close()
        db.close()
from fastapi import Response

@app.get("/logs")
def get_logs():
    try:
        with open("api_usage.log", "r", encoding="utf-8") as f:
            lines = f.readlines()[-100:]  # Last 100 lines only
        # Format nicely for humans
        formatted = ""
        for line in reversed(lines):
            if "INFO" not in line or "{" not in line:
                continue
            try:
                import json
                data = json.loads(line.split("| INFO | ", 1)[1])
                time = data.get("timestamp", "")[:19].replace("T", " ")
                ip = data.get("ip", "unknown")
                method = data.get("method", "")
                endpoint = data.get("endpoint", "")
                status = data.get("status", 0)
                duration = data.get("duration_ms", 0)

                status_color = "text-green-600" if status == 200 else "text-red-600"
                method_color = "text-blue-600 font-bold"

                formatted += f"""
                <div class="py-3 border-b border-gray-700 hover:bg-gray-800 transition">
                  <span class="text-gray-400 text-xs">{time}</span>
                  <span class="ml-4 text-cyan-400">{ip}</span>
                  <span class="ml-4 {method_color}">{method}</span>
                  <span class="ml-4 text-purple-400">{endpoint}</span>
                  <span class="ml-4 {status_color} font-bold">● {status}</span>
                  <span class="ml-4 text-yellow-400 text-sm">{duration:.0f}ms</span>
                </div>
                """
            except:
                continue
        return Response(content=formatted or "<div class='text-gray-500 p-8 text-center'>No API calls yet</div>", media_type="text/html")
    except:
        return Response(content="<div class='text-red-500 p-8 text-center'>Log file not found</div>", media_type="text/html")
    from fastapi import File, UploadFile, HTTPException
from fastapi import File, UploadFile, HTTPException
import pandas as pd 
import pandas as pd
from sqlalchemy import create_engine   
@app.post("/upload_csv")
async def upload_csv(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed")

    try:
        df = pd.read_csv(file.file)
        df.rename(columns={
    'Marital status': 'marital_status',
    'Application mode': 'application_mode',
    'Application order': 'application_order',
    'Course': 'course',
    'Daytime/evening attendance': 'attendance_regime',
    'Previous qualification': 'previous_qualification',
    'Nacionality': 'nationality',
    "Mother's qualification": 'mother_qualification',
    "Father's qualification": 'father_qualification',
    "Mother's occupation": 'mother_occupation',
    "Father's occupation": 'father_occupation',
    'Displaced': 'displaced',
    'Educational special needs': 'educational_special_needs',
    'Debtor': 'debtor',
    'Tuition fees up to date': 'tuition_fees_up_to_date',
    'Gender': 'gender',
    'Scholarship holder': 'scholarship_holder',
    'Age at enrollment': 'age_at_enrollment',
    'International': 'international',
    'Curricular units 1st sem (credited)': 'cu_1st_sem_credited',
    'Curricular units 1st sem (enrolled)': 'cu_1st_sem_enrolled',
    'Curricular units 1st sem (evaluations)': 'cu_1st_sem_evaluations',
    'Curricular units 1st sem (approved)': 'cu_1st_sem_approved',
    'Curricular units 1st sem (grade)': 'cu_1st_sem_grade',
    'Curricular units 1st sem (without evaluations)': 'cu_1st_sem_without_evaluation',
    'Curricular units 2nd sem (credited)': 'cu_2nd_sem_credited',
    'Curricular units 2nd sem (enrolled)': 'cu_2nd_sem_enrolled',
    'Curricular units 2nd sem (evaluations)': 'cu_2nd_sem_evaluations',
    'Curricular units 2nd sem (approved)': 'cu_2nd_sem_approved',
    'Curricular units 2nd sem (grade)': 'cu_2nd_sem_grade',
    'Curricular units 2nd sem (without evaluations)': 'cu_2nd_sem_without_evaluation',
    'Unemployment rate': 'unemployment_rate',
    'Inflation rate': 'inflation_rate',
    'GDP': 'gdp',
    'Target': 'target'
        }, inplace=True)

        # Connect to MySQL
        engine = create_engine("mysql+pymysql://root:1234@localhost/student_db")

        # Insert into table
        df .to_sql('student_data', con=engine, if_exists='append', index=False)

        return {
            "status": "success",
            "uploaded_students": len(df),
            "message": f"Successfully uploaded {len(df)} students!"
        }

    except HTTPException:
        raise
    except Exception as e:
        if 'db' in locals():
            db.rollback()
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'db' in locals():
            db.close()