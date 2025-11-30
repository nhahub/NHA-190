# mysql_pipeline.py
import mysql.connector
import pandas as pd
import joblib
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def create_tables():
    cnx = mysql.connector.connect(
        host=os.getenv('DB_HOST', 'localhost'),
        user=os.getenv('DB_USER', 'student_user'),
        password=os.getenv('DB_PASSWORD', 'your_secure_password'),
        database=os.getenv('DB_NAME', 'student_retention')
    )
    cursor = cnx.cursor()
    
    # Drop tables in reverse order (predictions first, then raw data)
    cursor.execute("DROP TABLE IF EXISTS predictions")
    cursor.execute("DROP TABLE IF EXISTS raw_students_data")
    
    # Create raw data table (ONLY 19 enrollment features + id + target)
    cursor.execute("""
        CREATE TABLE raw_students_data (
            id INT AUTO_INCREMENT PRIMARY KEY,
            marital_status INT,
            application_mode INT,
            application_order INT,
            course INT,
            daytime_evening_attendance INT,
            previous_qualification INT,
            nacionality INT,
            mothers_qualification INT,
            fathers_qualification INT,
            mothers_occupation INT,
            fathers_occupation INT,
            displaced INT,
            educational_special_needs INT,
            debtor INT,
            tuition_fees_up_to_date INT,
            gender INT,
            scholarship_holder INT,
            age_at_enrollment INT,
            international INT,
            target VARCHAR(20)
        )
    """)
    
    # Create predictions table
    cursor.execute("""
        CREATE TABLE predictions (
            id INT AUTO_INCREMENT PRIMARY KEY,
            student_id INT,
            dropout_probability FLOAT,
            risk_level VARCHAR(20),
            prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (student_id) REFERENCES raw_students_data(id)
        )
    """)
    
    cnx.commit()
    cnx.close()
    print("MySQL tables created") 

# Add this to the top of load_sample_data()
df = pd.read_csv('data/student_sample.csv')
print("CSV columns:", df.columns.tolist())
print("Number of columns:", len(df.columns))

def load_sample_data():
    """Load sample data into MySQL for testing"""
    df = pd.read_csv('data/student_sample.csv')
    
    # Map CSV column names to MySQL column names (snake_case)
    column_mapping = {
        'Marital status': 'marital_status',
        'Application mode': 'application_mode',
        'Application order': 'application_order',
        'Course': 'course',
        'Daytime/evening attendance': 'daytime_evening_attendance',
        'Previous qualification': 'previous_qualification',
        'Nacionality': 'nacionality',
        "Mother's qualification": 'mothers_qualification',
        "Father's qualification": 'fathers_qualification',
        "Mother's occupation": 'mothers_occupation',
        "Father's occupation": 'fathers_occupation',
        'Displaced': 'displaced',
        'Educational special needs': 'educational_special_needs',
        'Debtor': 'debtor',
        'Tuition fees up to date': 'tuition_fees_up_to_date',
        'Gender': 'gender',
        'Scholarship holder': 'scholarship_holder',
        'Age at enrollment': 'age_at_enrollment',
        'International': 'international',
        'Target': 'target'
    }
    
    # Select ONLY the 19 enrollment features + Target
    enrollment_columns = list(column_mapping.keys())
    df_filtered = df[enrollment_columns].copy()
    
    # Rename columns to match MySQL snake_case
    df_filtered = df_filtered.rename(columns=column_mapping)
    
    # Add ID column FIRST (not last!)
    df_filtered.insert(0, 'id', range(1, len(df_filtered) + 1))
    
    # Explicitly specify column order to match SQL
    mysql_columns = [
        'id', 'marital_status', 'application_mode', 'application_order', 'course',
        'daytime_evening_attendance', 'previous_qualification', 'nacionality',
        'mothers_qualification', 'fathers_qualification', 'mothers_occupation',
        'fathers_occupation', 'displaced', 'educational_special_needs', 'debtor',
        'tuition_fees_up_to_date', 'gender', 'scholarship_holder', 'age_at_enrollment',
        'international', 'target'
    ]
    
    cnx = mysql.connector.connect(
        host=os.getenv('DB_HOST', 'localhost'),
        user=os.getenv('DB_USER', 'student_user'),
        password=os.getenv('DB_PASSWORD', 'your_secure_password'),
        database=os.getenv('DB_NAME', 'student_retention')
    )
    
    cursor = cnx.cursor()
    for _, row in df_filtered[mysql_columns].iterrows():
        cursor.execute("""
            INSERT INTO raw_students_data 
            (id, marital_status, application_mode, application_order, course, 
             daytime_evening_attendance, previous_qualification, nacionality, 
             mothers_qualification, fathers_qualification, mothers_occupation, 
             fathers_occupation, displaced, educational_special_needs, debtor, 
             tuition_fees_up_to_date, gender, scholarship_holder, age_at_enrollment, 
             international, target)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, tuple(row))
    
    cnx.commit()
    cnx.close()
    print("Sample data loaded into MySQL")

def run_prediction_pipeline():
    """Main pipeline: MySQL → Preprocessing → Model → MySQL"""
    # 1. Load data from MySQL
    cnx = mysql.connector.connect(
        host=os.getenv('DB_HOST', 'localhost'),
        user=os.getenv('DB_USER', 'student_user'),
        password=os.getenv('DB_PASSWORD', 'your_secure_password'),
        database=os.getenv('DB_NAME', 'student_retention')
    )
    df = pd.read_sql("SELECT * FROM raw_students_data", cnx)
    cnx.close()
    
    # 2. Apply preprocessing
    print("Preprocessing data...")
    enrollment_features = [
        'marital_status', 'application_mode', 'application_order', 'course',
        'daytime_evening_attendance', 'previous_qualification', 'nacionality',
        'mothers_qualification', 'fathers_qualification', 'mothers_occupation',
        'fathers_occupation', 'displaced', 'educational_special_needs', 'debtor',
        'tuition_fees_up_to_date', 'gender', 'scholarship_holder', 'age_at_enrollment',
        'international'
    ]
    
    # Handle missing values
    df = df.fillna(df.median(numeric_only=True))
    
    # Create derived features
    df['financial_risk_flag'] = ((df['debtor'] == 1) & (df['tuition_fees_up_to_date'] == 0)).astype(int)
    
    # Convert to numpy array to avoid feature name warnings
    X = df[enrollment_features].values
    
    # 3. Load and apply model
    print("Loading model and making predictions...")
    model = joblib.load('models/dropout_model_v1.0_20251112_202951.pkl')
    probabilities = model.predict_proba(X)[:, 1]
    risk_levels = ['High' if p > 0.7 else 'Medium' if p > 0.3 else 'Low' for p in probabilities]
    
    # 4. Save predictions to MySQL
    cnx = mysql.connector.connect(
        host=os.getenv('DB_HOST', 'localhost'),
        user=os.getenv('DB_USER', 'student_user'),
        password=os.getenv('DB_PASSWORD', 'your_secure_password'),
        database=os.getenv('DB_NAME', 'student_retention')
    )
    cursor = cnx.cursor()
    
    for i, (prob, risk) in enumerate(zip(probabilities, risk_levels)):
        cursor.execute(
            """
            INSERT INTO predictions (student_id, dropout_probability, risk_level)
            VALUES (%s, %s, %s)
            """,
            (int(df.iloc[i]['id']), float(prob), str(risk))
        )
    
    cnx.commit()
    cnx.close()
    print("Predictions saved to MySQL")
    print(f"High-risk students: {risk_levels.count('High')}")
    # After running predictions, add this to verify
    print("\n=== PREDICTION VERIFICATION ===")
    for i in range(len(df)):
        student_id = int(df.iloc[i]['id'])
        actual_target = df.iloc[i]['target']
        predicted_probability = probabilities[i]
        predicted_risk_level = risk_levels[i]
        
        print(f"Student {student_id}:")
        print(f"  Actual: {actual_target}")
        print(f"  Predicted Probability: {predicted_probability:.4f}")
        print(f"  Predicted Risk Level: {predicted_risk_level}")
        print(f"  Correct? {'YES' if (actual_target == 'Dropout' and predicted_risk_level == 'High') or (actual_target == 'Graduate' and predicted_risk_level != 'High') else 'NO'}")
        print()
    
if __name__ == "__main__":
    create_tables()
    load_sample_data()
    run_prediction_pipeline()
    