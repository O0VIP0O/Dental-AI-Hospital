import sqlite3

def get_bach():
    try:
        conn = sqlite3.connect("dental_project_DB.db")
        cursor = conn.cursor()
        cursor.execute(
            "SELECT batch_id from batch ORDER by batch_id desc LIMIT 1;",
        )
        batch = cursor.fetchone()[0]
        cursor.close()
        conn.close()

       
        return batch
    except sqlite3.Error as e:
        print(f"Database error: {e}")




def insert_cases(name, age, gender, phone):
    try:
        conn = sqlite3.connect("dental_project_DB.db")
        cursor = conn.cursor()
        batch = get_bach()
        cursor.execute(
            """
            INSERT INTO cases (batch_id, name, age, gender, phone)
            VALUES (?, ?, ? , ?, ?);
            """,
            (batch , name, age, gender, phone)
        )
        conn.commit()

        case_id = cursor.lastrowid

        cursor.execute(
            """
            INSERT INTO student_department_cases (department_id, case_id, appointment_date, appointment_time)
            VALUES ('D001', ?, DATE('now', 'localtime'), TIME('now', 'localtime')); 
            """,
            (case_id, )
        )
        conn.commit()

        cursor.close()
        conn.close()

    except sqlite3.Error as e:
        print(f"Database error: {e}")


def get_patient_Queue():
    """Fetch all patients without a student and return JSON keyed by patient name."""
    try:
        conn = sqlite3.connect("dental_project_DB.db")
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("""
            SELECT 
                c.case_id,
                c.name,
                c.phone,
                c.age,
                c.gender,
                s.appointment_date,
                s.description
            FROM student_department_cases AS s
            JOIN cases AS c ON s.case_id = c.case_id
            WHERE s.student_id IS NULL;
        """)
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        
        patient_dict = {}
        for row in rows:
            key = row['name'].lower().replace(' ', '-')  # e.g., 'mike-chen'
            patient_dict[key] = {
                "id": str(row['case_id']),
                "phone": row['phone'] or "",
                "age": row['age'] or "",
                "gender": row['gender'] or "",
                "appointment_date": row['appointment_date'] or "",
                "description": row['description'] or ""
            }
        
        if patient_dict:
            return {"success": True, "data": patient_dict}
        else:
            return {"success": False, "message": "No patients found"}
    
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return {"success": False, "message": "Server error, please try again"}


def update_studentID_for_case(student_id, case_id, department_id):
    try:
        conn = sqlite3.connect("dental_project_DB.db")
        cursor = conn.cursor()

        cursor.execute("""update student_department_cases
                            set student_id = ?
                            where case_id = ? and department_id = ?;""",
                              (student_id, case_id, department_id))
        conn.commit()
        cursor.close()
        conn.close()

    except sqlite3.Error as e:
        print(f"Database Error : {e}")