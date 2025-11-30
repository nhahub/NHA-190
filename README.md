
***

# Student Retention Prediction System – Project Setup & Team Guide

## Project Overview
You are working with a MySQL-based system that predicts student dropout risk using machine learning. The project provides:
- A trained ML model, pipeline, and sample data
- A FastAPI endpoint for prediction queries
- MySQL-managed data storage for student info and predictions

## Quick Setup (For All Team Members)

### 1. Install MySQL
```bash
sudo apt update
sudo apt install mysql-server mysql-client
sudo mysql_secure_installation
sudo systemctl start mysql
sudo systemctl enable mysql
```
Set a root password during MySQL secure installation.

### 2. Prepare the Database
```bash
sudo mysql -u root -p
CREATE DATABASE student_retention;
CREATE USER 'student_user'@'localhost' IDENTIFIED BY 'your_secure_password';
GRANT ALL PRIVILEGES ON student_retention.* TO 'student_user'@'localhost';
FLUSH PRIVILEGES;
EXIT;
```

### 3. Clone & Set Up Python Environment
```bash
git clone https://github.com/ahmedSaf412/student-retention-mysql.git
cd student-retention-mysql
python3 -m venv mysql_venv
source mysql_venv/bin/activate
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Create a `.env` file in the project root:
```
DB_HOST=localhost
DB_USER=student_user
DB_PASSWORD=your_secure_password
DB_NAME=student_retention
```

### 5. Run the Prediction Pipeline
```bash
python mysql_pipeline.py
```
Output should confirm successful sample data loading and prediction saving.

***

## Project Structure

- `mysql_pipeline.py`: Main pipeline (run predictions)
- `api/app.py`: FastAPI endpoint (add features as needed)
- `data/student_sample.csv`: Sample student data
- `models/`: Pre-trained ML models *(DO NOT MODIFY)*
- `data_Ingestion/`: Custom scripts for data ingestion
- `dashboard/app.py`: Streamlit dashboard (to build)
- `README.md`: Project readme and setup notes
- `test_db.py`: Script to test DB connection (optional)

***

## Team Tasks & Further Improvements

### Data Engineer: Database Tuning
- Add indexes for query speed:
  - `predictions(risk_level)`, `predictions(student_id)`, `predictions(prediction_date)`
- Add constraints for data quality:
  - Example: `CHECK (dropout_probability BETWEEN 0 AND 1)`

### Data Engineer: Daily Ingestion Script
- Build `data_Ingestion/daily_sampler.py` to create fresh daily student batches.
- Integrate this batch feed into the pipeline.

### Backend Developer: FastAPI Enhancement
- Extend `api/app.py` with endpoints:
  - `/high_risk_students`
  - `/predictions/today`
  - `/predict_batch`
- Add input validation & error handling.
- Document API with OpenAPI/Swagger.

### DevOps: Automate Prediction Runs
- Install & initialize Apache Airflow.
- Build a DAG in `workflow/student_pipeline_dag.py` to automate daily pipeline runs.
- Start Airflow webserver and scheduler.

### BI Analyst: Dashboard Creation
- Install Streamlit.
- Build a dashboard in `dashboard/app.py` to visualize predictions and alerts.

***

## Critical Notes
- Do **NOT** modify models in `models/` – these are production-ready.
- Always activate your environment:  
  `source mysql_venv/bin/activate`
- Test each component individually before full integration.
- Sample data includes two high-risk students for debugging and demo.

***

## Quick Start Commands

```bash
python mysql_pipeline.py                 # Run pipeline & save predictions
uvicorn api.app:app --reload --port 8000 # Start FastAPI
curl http://localhost:8000/predict/1     # Test FastAPI endpoint
airflow webserver --port 8080            # Airflow UI (after setup)
airflow scheduler                        # Airflow job runner
streamlit run dashboard/app.py           # Launch dashboard (after setup)
py -m streamlit run dashboard/app.py     # Launch dashboard (Windows alternative)
```

## Dashboard
To run the interactive dashboard:
1. Ensure database is set up.
2. Run: `py -m streamlit run dashboard/app.py`

***

