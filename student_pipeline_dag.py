"""
Student Retention Prediction - Airflow DAG
Simple daily pipeline to run predictions
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import pipeline functions
from mysql_pipeline import create_tables, load_sample_data, run_prediction_pipeline

# Default arguments for the DAG
default_args = {
    'owner': 'data-engineering',
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'start_date': days_ago(1),
    'email_on_failure': False,
    'email_on_retry': False,
}

# Create the DAG
dag = DAG(
    'student_retention_prediction_dag',
    default_args=default_args,
    description='Daily student dropout prediction pipeline',
    schedule_interval='0 2 * * *',  # Run at 2 AM daily
    catchup=False,
    tags=['student-retention', 'ml-pipeline'],
)

# Define task functions
def init_db():
    """Initialize database tables"""
    print("🔄 Initializing database...")
    create_tables()
    print("✅ Database initialized")

def load_data():
    """Load sample data into database"""
    print("🔄 Loading sample data...")
    load_sample_data()
    print("✅ Sample data loaded")

def run_predictions():
    """Run prediction pipeline"""
    print("🔄 Running predictions...")
    run_prediction_pipeline()
    print("✅ Predictions completed")

# Create tasks
task_init_db = PythonOperator(
    task_id='init_database',
    python_callable=init_db,
    dag=dag,
)

task_load_data = PythonOperator(
    task_id='load_sample_data',
    python_callable=load_data,
    dag=dag,
)

task_run_predictions = PythonOperator(
    task_id='run_predictions',
    python_callable=run_predictions,
    dag=dag,
)

# Set task dependencies
task_init_db >> task_load_data >> task_run_predictions