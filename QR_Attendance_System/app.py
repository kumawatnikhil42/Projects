import qrcode
import smtplib
from flask import Flask, render_template, redirect, url_for, flash, session, request, send_from_directory, Response
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
from dotenv import load_dotenv
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import cv2
import csv
import threading
import time
import sqlite3
from threading import Lock
import os
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# Secret key for session management
app.secret_key = os.getenv('SECRET_KEY', 'your_secret_key_here')

# SQLite configuration (database file path)
DATABASE = 'database.db'

# Lock for attendance file writes
attendance_lock = Lock()

def get_db_connection():
    """Establish a connection to the SQLite database."""
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

def send_email(to_email, subject, body, attachment_path):
    """Function to send an email with an attachment."""
    smtp_server = os.getenv('SMTP_SERVER')
    smtp_port = int(os.getenv('SMTP_PORT', '587'))
    smtp_user = os.getenv('SMTP_USER')
    smtp_password = os.getenv('SMTP_PASSWORD')

    msg = MIMEMultipart()
    msg['From'] = smtp_user
    msg['To'] = to_email
    msg['Subject'] = subject

    msg.attach(MIMEText(body, 'plain'))

    # Attach the QR code image if provided
    if attachment_path:
        try:
            with open(attachment_path, 'rb') as attachment:
                part = MIMEBase('application', 'octet-stream')
                part.set_payload(attachment.read())
                encoders.encode_base64(part)
                part.add_header('Content-Disposition', f'attachment; filename="{os.path.basename(attachment_path)}"')
                msg.attach(part)
        except Exception as e:
            flash(f'Failed to attach the file: {str(e)}', 'danger')

    try:
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()  # Upgrade to a secure connection
            server.login(smtp_user, smtp_password)
            server.send_message(msg)
            # flash('QR code sent to your email!', 'success')
    except Exception as e:
        flash(f'Failed to send email: {str(e)}', 'danger')

@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        # Fetch user from database
        conn = get_db_connection()
        cursor = conn.execute("SELECT * FROM users WHERE email = ?", (email,))
        user = cursor.fetchone()
        conn.close()

        if user and check_password_hash(user['password'], password):  # Check password hash
            session['user_id'] = user['id']
            flash('Login successful', 'success')
            return redirect(url_for('scanner'))
        else:
            flash('Invalid email or password', 'danger')

    return render_template('login.html')
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

def share_sheet_with_user(email):
    """Share the Google Sheet with the specified email address."""
    try:
        client = authenticate_google_sheets()
        sheet = client.open("Attendanceqr")
        
        # Build the Drive API service
        drive_service = build('drive', 'v3', credentials=client.auth.credentials)

        # Prepare the permission request
        permission_body = {
            'type': 'user',
            'role': 'reader',
            'emailAddress': email
        }

        # Create permission
        permission = drive_service.permissions().create(
            fileId=sheet.id,
            body=permission_body,
            fields='id',
            sendNotificationEmail=True  # Set to True to send a notification email
        ).execute()
        
        print(f"Shared sheet with {email}")
    except HttpError as error:
        print(f"An error occurred: {error}")
    except Exception as e:
        print(f"Error sharing the sheet: {e}")

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        # Password validation
        if password != confirm_password:
            flash('Passwords do not match', 'danger')
            return redirect(url_for('register'))

        hashed_password = generate_password_hash(password)
        created_at = datetime.now()

        # Generate QR code
        qr_data = f"{name},{email}"
        qr_image_filename = f"qr_codes/{name}_{email}.png"
        qr_image_path = os.path.join('static', qr_image_filename)
        qr = qrcode.make(qr_data)
        os.makedirs(os.path.dirname(qr_image_path), exist_ok=True)
        qr.save(qr_image_path)

        # Insert into SQLite
        conn = get_db_connection()
        conn.execute("INSERT INTO users (name, email, password, registration_date, qr_code) VALUES (?, ?, ?, ?, ?) ",
                     (name, email, hashed_password, created_at, qr_image_filename))
        conn.commit()
        conn.close()

        # Send QR code via email
        subject = 'Your QR Code Registration'
        body = f"Hello {name},\n\nThank you for registering! Here is your QR code.\n\nBest regards,\nYour Company"
        send_email(email, subject, body, qr_image_path)
        share_sheet_with_user(email)
        flash('Registration successful! You can now log in.', 'success')
        return redirect(url_for('login'))

    return render_template('register.html')

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

@app.route('/static/qr_codes/<path:filename>')
def serve_qr_code(filename):
    return send_from_directory('static/qr_codes', filename)

def generate_frames():
    # Reduce the camera resolution for smoother streaming
    # camera = cv2.VideoCapture("rtsp://admin:L2D24ABD@192.168.1.42:554/cam/realmonitor?channel=1&subtype=0&unicast=true&proto=onvif")
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Error: Unable to open camera")
        return

    # Lower resolution to improve performance
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    # Limit the frame rate
    fps_limit = 10
    prev_time = time.time()
    
    detector = cv2.QRCodeDetector()
    checked_in_users = {}
    attendance_status = None

    while True:
        success, frame = camera.read()
        current_time = time.time()

        # Only process frame if time elapsed from the last frame exceeds fps_limit interval
        if success and (current_time - prev_time) > 1.0 / fps_limit:
            prev_time = current_time

            # QR Code detection and processing
            data, points, _ = detector.detectAndDecode(frame)
            if data:
                name_email = data.split(',')
                if len(name_email) >= 2:
                    name = name_email[0]
                    email = name_email[1]

                    conn = get_db_connection()
                    cursor = conn.execute("SELECT * FROM users WHERE name = ? AND email = ?", (name, email))
                    user = cursor.fetchone()
                    conn.close()

                    if user:
                        if email in checked_in_users:
                            attendance_status = 'You will check out after 1 hour!'
                        else:
                            mark_attendance(name, email, "checkin")
                            checked_in_users[email] = True
                            attendance_status = f'Scan successful! Attendance marked for {name}!'
                            schedule_checkout(name, email)
                    else:
                        attendance_status = 'User not found!'
                else:
                    attendance_status = 'Invalid QR code data!'

            # Overlay attendance_status on the frame
            if attendance_status:
                cv2.putText(frame, attendance_status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                attendance_status = None  # Reset the status after displaying

            # Encode the frame
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    camera.release()

@app.route('/scanner')
def scanner():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('scanner.html')

def authenticate_google_sheets():
    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive.file",
        "https://www.googleapis.com/auth/drive"
    ]
    creds = ServiceAccountCredentials.from_json_keyfile_name('cred.json', scope)
    client = gspread.authorize(creds)
    return client

def send_attendance_email(email, name, status):
    """Send an email to the user confirming their attendance status."""
    subject = f'Attendance {status.capitalize()} Confirmation'
    body = (
        f"Hello {name},\n\n"
        f"Your attendance has been marked as {status}.\n\n"
        f"You can view your attendance record in the following Google Sheet:\n"
        f"[View Attendance Sheet](https://docs.google.com/spreadsheets/d/1LCEQk5ZcyMulXfTWfty7ZpRhbHNC2CKgSaiSGSGUuXI/edit?usp=sharing)\n\n"
        f"Best regards,\n"
        f"Your Company"
    )
    send_email(email, subject, body, None)  # No attachment needed

def mark_attendance(name, email, status):
    """Mark attendance in Google Sheets."""
    client = authenticate_google_sheets()
    sheet = client.open("Attendanceqr").sheet1  # Open the Google Sheet and select the first sheet

    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    time_now = now.strftime("%H:%M:%S")

    row = [name, email, date, time_now, status]
    
    # Append the row to Google Sheets
    try:
        print("Google Sheets authentication successful.")
        sheet.append_row(row)
        print(f"Attendance recorded for {name} ({status}) in Google Sheets")
        send_attendance_email(email,name,status)
    except Exception as e:
        print(f"Error writing to Google Sheets: {e}")

def schedule_checkout(name, email):
    """Schedules a checkout for the user after 1 hour and sends email confirmation."""
    def check_out():
        time.sleep(3600)  # Wait for 1 hour
        mark_attendance(name, email, "checkout")  # Mark checkout
        print(f"Checked out user {name} after 1 hour.")

    threading.Thread(target=check_out, daemon=True).start()



@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    os.makedirs('static/qr_codes', exist_ok=True)

    # Initialize the database with the necessary tables if not already set up
    conn = get_db_connection()
    conn.execute('''CREATE TABLE IF NOT EXISTS users (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT NOT NULL,
                        email TEXT UNIQUE NOT NULL,
                        password TEXT NOT NULL,
                        registration_date TEXT NOT NULL,
                        qr_code TEXT
                    )''')
    conn.close()

    # app.run(debug=True)
    app.run(host='0.0.0.0', port=5000,debug=True)

