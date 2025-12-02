import pandas as pd
from sqlalchemy import create_engine

# Load CSV
df = pd.read_csv(r"C:\ProgramData\MySQL\MySQL Server 8.0\Uploads\dataset_labeled.csv")

# Rename columns to match MySQL table
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
df.to_sql('student_data', con=engine, if_exists='append', index=False)
