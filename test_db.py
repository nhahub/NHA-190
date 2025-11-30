import mysql.connector

try:
    # Connect using the application user
    cnx = mysql.connector.connect(
        host='localhost',
        user='student_user',
        password='Stud_usr_123',
        database='student_retention'
    )
    print("✅ Successfully connected to MySQL!")
    
    # Test creating a cursor
    cursor = cnx.cursor()
    cursor.execute("SELECT DATABASE();")
    db_name = cursor.fetchone()
    print(f"Connected to database: {db_name[0]}")
    
    cnx.close()
    
except mysql.connector.Error as err:
    print(f"❌ Database error: {err}")
except Exception as err:
    print(f"❌ Other error: {err}")