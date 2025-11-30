import streamlit as st
import pandas as pd
import mysql.connector
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page Config
st.set_page_config(
    page_title="Student Retention Dashboard",
    page_icon="🎓",
    layout="wide"
)

# Database Connection
@st.cache_resource
def get_db_connection():
    return mysql.connector.connect(
        host=os.getenv('DB_HOST', 'localhost'),
        user=os.getenv('DB_USER', 'student_user'),
        password=os.getenv('DB_PASSWORD', 'your_secure_password'),
        database=os.getenv('DB_NAME', 'student_retention')
    )

def load_data():
    conn = get_db_connection()
    query = """
    SELECT 
        p.student_id,
        p.dropout_probability,
        p.risk_level,
        p.prediction_date,
        r.course,
        r.daytime_evening_attendance,
        r.previous_qualification,
        r.nacionality,
        r.age_at_enrollment,
        r.gender
    FROM predictions p
    JOIN raw_students_data r ON p.student_id = r.id
    ORDER BY p.prediction_date DESC
    """
    df = pd.read_sql(query, conn)
    conn.close()
    return df

# Title
st.title("🎓 Student Retention Prediction Dashboard")
st.markdown("Monitor student dropout risks and take proactive measures.")

try:
    # Load Data
    with st.spinner('Loading data from database...'):
        df = load_data()

    if df.empty:
        st.warning("No prediction data found in the database. Please run the pipeline first.")
    else:
        # Metrics Section
        st.subheader("📊 Key Metrics")
        col1, col2, col3, col4 = st.columns(4)
        
        total_students = len(df)
        high_risk_count = len(df[df['risk_level'] == 'High'])
        medium_risk_count = len(df[df['risk_level'] == 'Medium'])
        avg_prob = df['dropout_probability'].mean()

        col1.metric("Total Students", total_students)
        col2.metric("High Risk Students", high_risk_count, delta_color="inverse")
        col3.metric("Medium Risk Students", medium_risk_count, delta_color="off")
        col4.metric("Avg Dropout Probability", f"{avg_prob:.2%}")

        st.divider()

        # Charts Section
        col_chart1, col_chart2 = st.columns(2)

        with col_chart1:
            st.subheader("Risk Level Distribution")
            risk_counts = df['risk_level'].value_counts()
            st.bar_chart(risk_counts, color=["#FF4B4B", "#FFA500", "#00CC96"] if len(risk_counts) == 3 else None)

        with col_chart2:
            st.subheader("Dropout Probability Distribution")
            st.bar_chart(df['dropout_probability'])

        st.divider()

        # High Risk Alert Section
        st.subheader("🚨 High Risk Students (Action Required)")
        high_risk_df = df[df['risk_level'] == 'High'].sort_values(by='dropout_probability', ascending=False)
        
        if not high_risk_df.empty:
            st.dataframe(
                high_risk_df[['student_id', 'dropout_probability', 'risk_level', 'course', 'age_at_enrollment']],
                use_container_width=True,
                column_config={
                    "dropout_probability": st.column_config.ProgressColumn(
                        "Probability",
                        format="%.2f",
                        min_value=0,
                        max_value=1,
                    ),
                }
            )
        else:
            st.success("No high-risk students detected! 🎉")

        # Full Data Browser
        with st.expander("📂 View All Student Data"):
            st.dataframe(df)

except Exception as e:
    st.error(f"Error connecting to database: {e}")
    st.info("Ensure MySQL is running and the .env file is configured correctly.")
