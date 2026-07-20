import pickle
import numpy as np
import mysql.connector
from mysql.connector import Error
from datetime import datetime

DB_PASSWORD = "root"


def get_db_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password=DB_PASSWORD,
        database="classroom_presence"
    )


def init_db():
    try:
        conn = mysql.connector.connect(host="localhost", user="root", password=DB_PASSWORD)
        cursor = conn.cursor()
        cursor.execute("CREATE DATABASE IF NOT EXISTS classroom_presence")
        conn.close()

        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INT AUTO_INCREMENT PRIMARY KEY,
                username VARCHAR(50) NOT NULL UNIQUE,
                email VARCHAR(100) NOT NULL UNIQUE,
                password_hash VARCHAR(255) NOT NULL,
                face_encoding LONGBLOB,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS students (
                student_id VARCHAR(50) PRIMARY KEY,
                last_name VARCHAR(100),
                first_name VARCHAR(100),
                face_encoding LONGBLOB,
                photo_path VARCHAR(255)
            )
        ''')
        for col_sql in [
            "ALTER TABLE students ADD COLUMN face_encoding LONGBLOB",
            "ALTER TABLE students ADD COLUMN photo_path VARCHAR(255)",
        ]:
            try:
                cursor.execute(col_sql)
                conn.commit()
            except Error:
                pass

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS admins (
                id INT AUTO_INCREMENT PRIMARY KEY,
                username VARCHAR(50) NOT NULL UNIQUE,
                password_hash VARCHAR(255) NOT NULL,
                face_encoding_blob LONGBLOB
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS course_schedules (
                course_code VARCHAR(50) PRIMARY KEY,
                days VARCHAR(50),
                time_window VARCHAR(100),
                room VARCHAR(50)
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS enrollments (
                student_id VARCHAR(50),
                course_code VARCHAR(50),
                PRIMARY KEY (student_id, course_code),
                FOREIGN KEY (student_id) REFERENCES students(student_id) ON DELETE CASCADE,
                FOREIGN KEY (course_code) REFERENCES course_schedules(course_code) ON DELETE CASCADE
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS presence_logs (
                id INT AUTO_INCREMENT PRIMARY KEY,
                timestamp DATETIME,
                log_type VARCHAR(50),
                detected_student_id VARCHAR(50),
                snapshot_path VARCHAR(255),
                message TEXT,
                FOREIGN KEY (detected_student_id) REFERENCES students(student_id) ON DELETE SET NULL
            )
        ''')

        conn.commit()
        print("Database initialized successfully.")
    except Error as e:
        print(f"Error: {e}")
    finally:
        if 'conn' in locals() and conn.is_connected():
            cursor.close()
            conn.close()


def get_admin_count():
    """Return number of registered admins."""
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM users")
        return cursor.fetchone()[0]
    except Error:
        return 0
    finally:
        if conn and conn.is_connected():
            cursor.close()
            conn.close()


def create_user(username, email, password_hash):
    """Insert a new admin. Returns (True, None) on success or (False, error_msg)."""
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO users (username, email, password_hash) VALUES (%s, %s, %s)",
            (username, email, password_hash)
        )
        conn.commit()
        return True, None
    except Error as e:
        if e.errno == 1062:
            msg = "Username already taken." if "username" in str(e).lower() else "Email already registered."
            return False, msg
        return False, f"Database error: {e}"
    finally:
        if conn and conn.is_connected():
            cursor.close()
            conn.close()


def get_user_by_username(username):
    """Return admin row as dict or None."""
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
        return cursor.fetchone()
    except Error as e:
        print(f"Error fetching user: {e}")
        return None
    finally:
        if conn and conn.is_connected():
            cursor.close()
            conn.close()


def save_face_encoding(username, encoding):
    """Save face embedding blob for an admin user. Returns True if a row was updated."""
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        blob = pickle.dumps(np.array(encoding))
        cursor.execute(
            "UPDATE users SET face_encoding = %s WHERE username = %s",
            (blob, username)
        )
        conn.commit()
        return cursor.rowcount > 0
    except Error as e:
        print(f"Error saving face encoding: {e}")
        return False
    finally:
        if conn and conn.is_connected():
            cursor.close()
            conn.close()


def get_all_face_encodings():
    """Return list of (username, encoding array) for all admin users with face data."""
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT username, face_encoding FROM users WHERE face_encoding IS NOT NULL")
        results = []
        for username, blob in cursor.fetchall():
            encoding = np.array(pickle.loads(blob))
            results.append((username, encoding))
        return results
    except Error as e:
        print(f"Error loading face encodings: {e}")
        return []
    finally:
        if conn and conn.is_connected():
            cursor.close()
            conn.close()


def get_all_students():
    """Return all students with face status, photo, and enrolled courses."""
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("""
            SELECT s.student_id, s.last_name, s.first_name,
                   (s.face_encoding IS NOT NULL) AS has_face,
                   s.photo_path,
                   GROUP_CONCAT(e.course_code ORDER BY e.course_code SEPARATOR ',') AS enrolled_courses
            FROM students s
            LEFT JOIN enrollments e ON s.student_id = e.student_id
            GROUP BY s.student_id, s.last_name, s.first_name, s.face_encoding, s.photo_path
            ORDER BY s.last_name, s.first_name
        """)
        return cursor.fetchall()
    except Error as e:
        print(f"Error fetching students: {e}")
        return []
    finally:
        if conn and conn.is_connected():
            cursor.close()
            conn.close()


def save_student_face_encoding(student_id, encoding):
    """Save face embedding for a student."""
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        blob = pickle.dumps(np.array(encoding))
        cursor.execute(
            "UPDATE students SET face_encoding = %s WHERE student_id = %s",
            (blob, student_id)
        )
        conn.commit()
        return cursor.rowcount > 0
    except Error as e:
        print(f"Error saving student face: {e}")
        return False
    finally:
        if conn and conn.is_connected():
            cursor.close()
            conn.close()


def save_student_photo_path(student_id, photo_path):
    """Save file path of student's reference photo."""
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE students SET photo_path = %s WHERE student_id = %s",
            (photo_path, student_id)
        )
        conn.commit()
        return cursor.rowcount > 0
    except Error as e:
        print(f"Error saving photo path: {e}")
        return False
    finally:
        if conn and conn.is_connected():
            cursor.close()
            conn.close()


def get_all_student_face_encodings():
    """Return list of (student_id, full_name, encoding) for camera identification."""
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT student_id, CONCAT(first_name, ' ', last_name), face_encoding "
            "FROM students WHERE face_encoding IS NOT NULL"
        )
        results = []
        for student_id, name, blob in cursor.fetchall():
            encoding = np.array(pickle.loads(blob))
            results.append((student_id, name, encoding))
        return results
    except Error as e:
        print(f"Error loading student encodings: {e}")
        return []
    finally:
        if conn and conn.is_connected():
            cursor.close()
            conn.close()


def get_all_courses():
    """Return all courses with enrolled student count."""
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("""
            SELECT cs.course_code, cs.days, cs.time_window, cs.room,
                   COUNT(e.student_id) AS student_count
            FROM course_schedules cs
            LEFT JOIN enrollments e ON cs.course_code = e.course_code
            GROUP BY cs.course_code, cs.days, cs.time_window, cs.room
            ORDER BY cs.course_code
        """)
        return cursor.fetchall()
    except Error as e:
        print(f"Error fetching courses: {e}")
        return []
    finally:
        if conn and conn.is_connected():
            cursor.close()
            conn.close()


def add_course(course_code, days, time_window, room):
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO course_schedules (course_code, days, time_window, room) VALUES (%s,%s,%s,%s) "
            "ON DUPLICATE KEY UPDATE days=VALUES(days), time_window=VALUES(time_window), room=VALUES(room)",
            (course_code, days, time_window, room)
        )
        conn.commit()
        return True, None
    except Error as e:
        return False, str(e)
    finally:
        if conn and conn.is_connected():
            cursor.close()
            conn.close()


def delete_course(course_code):
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM course_schedules WHERE course_code = %s", (course_code,))
        conn.commit()
        return cursor.rowcount > 0
    except Error as e:
        print(f"Error deleting course: {e}")
        return False
    finally:
        if conn and conn.is_connected():
            cursor.close()
            conn.close()


def get_student_enrollments(student_id):
    """Return list of course dicts the student is enrolled in."""
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("""
            SELECT cs.course_code, cs.days, cs.time_window, cs.room
            FROM course_schedules cs
            JOIN enrollments e ON cs.course_code = e.course_code
            WHERE e.student_id = %s
            ORDER BY cs.course_code
        """, (student_id,))
        return cursor.fetchall()
    except Error as e:
        print(f"Error fetching enrollments: {e}")
        return []
    finally:
        if conn and conn.is_connected():
            cursor.close()
            conn.close()


def enroll_student(student_id, course_code):
    """Add a single enrollment. Returns (True, None) or (False, error)."""
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT IGNORE INTO enrollments (student_id, course_code) VALUES (%s, %s)",
            (student_id, course_code)
        )
        conn.commit()
        return True, None
    except Error as e:
        return False, str(e)
    finally:
        if conn and conn.is_connected():
            cursor.close()
            conn.close()


def unenroll_student(student_id, course_code):
    """Remove a single enrollment."""
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "DELETE FROM enrollments WHERE student_id = %s AND course_code = %s",
            (student_id, course_code)
        )
        conn.commit()
        return cursor.rowcount > 0
    except Error as e:
        print(f"Error removing enrollment: {e}")
        return False
    finally:
        if conn and conn.is_connected():
            cursor.close()
            conn.close()


def insert_students(rows):
    """Bulk insert students from CSV rows. Returns (inserted, skipped, errors)."""
    conn = None
    inserted = skipped = 0
    errors = []
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        for row in rows:
            try:
                cursor.execute(
                    "INSERT INTO students (student_id, last_name, first_name) VALUES (%s, %s, %s) "
                    "ON DUPLICATE KEY UPDATE last_name=VALUES(last_name), first_name=VALUES(first_name)",
                    (row["student_id"], row["last_name"], row["first_name"])
                )
                inserted += 1
            except Error as e:
                skipped += 1
                errors.append(str(e))
        conn.commit()
    except Error as e:
        errors.append(str(e))
    finally:
        if conn and conn.is_connected():
            cursor.close()
            conn.close()
    return inserted, skipped, errors


def insert_schedules(rows):
    """Bulk insert course schedules from CSV rows."""
    conn = None
    inserted = skipped = 0
    errors = []
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        for row in rows:
            try:
                cursor.execute(
                    "INSERT INTO course_schedules (course_code, days, time_window, room) VALUES (%s, %s, %s, %s) "
                    "ON DUPLICATE KEY UPDATE days=VALUES(days), time_window=VALUES(time_window), room=VALUES(room)",
                    (row["course_code"], row["days"], row["time_window"], row["room"])
                )
                inserted += 1
            except Error as e:
                skipped += 1
                errors.append(str(e))
        conn.commit()
    except Error as e:
        errors.append(str(e))
    finally:
        if conn and conn.is_connected():
            cursor.close()
            conn.close()
    return inserted, skipped, errors


def insert_enrollments(rows):
    """Bulk insert enrollments from CSV rows."""
    conn = None
    inserted = skipped = 0
    errors = []
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        for row in rows:
            try:
                cursor.execute(
                    "INSERT IGNORE INTO enrollments (student_id, course_code) VALUES (%s, %s)",
                    (row["student_id"], row["course_code"])
                )
                inserted += 1
            except Error as e:
                skipped += 1
                errors.append(str(e))
        conn.commit()
    except Error as e:
        errors.append(str(e))
    finally:
        if conn and conn.is_connected():
            cursor.close()
            conn.close()
    return inserted, skipped, errors


def _day_matches(schedule_days, today_code):
    """Check if today is in a schedule string like 'MWF' or 'TTH'."""
    s = schedule_days.upper().replace('TH', 'X')
    t = today_code.upper().replace('TH', 'X')
    return t in s


def _time_matches(time_window, current_time):
    """Check if current_time falls within '9:00-10:00' style window."""
    try:
        start_str, end_str = time_window.split('-')
        start = datetime.strptime(start_str.strip(), '%H:%M').time()
        end   = datetime.strptime(end_str.strip(),   '%H:%M').time()
        return start <= current_time <= end
    except Exception:
        return False


def get_active_schedule_for_student(student_id):
    """Return the course schedule dict if this student has a class right now, else None."""
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            """SELECT cs.course_code, cs.days, cs.time_window, cs.room
               FROM course_schedules cs
               JOIN enrollments e ON cs.course_code = e.course_code
               WHERE e.student_id = %s""",
            (student_id,)
        )
        schedules = cursor.fetchall()

        now = datetime.now()
        day_map = {0:'M', 1:'T', 2:'W', 3:'TH', 4:'F', 5:'S', 6:'SU'}
        today = day_map[now.weekday()]

        for sched in schedules:
            if _day_matches(sched['days'], today) and _time_matches(sched['time_window'], now.time()):
                return sched
        return None
    except Error as e:
        print(f"Error checking schedule: {e}")
        return None
    finally:
        if conn and conn.is_connected():
            cursor.close()
            conn.close()


def log_classroom_event(timestamp, log_type, student_id, snapshot_path, message):
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            '''INSERT INTO presence_logs (timestamp, log_type, detected_student_id, snapshot_path, message)
               VALUES (%s, %s, %s, %s, %s)''',
            (timestamp, log_type, student_id, snapshot_path, message)
        )
        conn.commit()
    except Error as e:
        print(f"Error logging event: {e}")
    finally:
        if conn and conn.is_connected():
            cursor.close()
            conn.close()