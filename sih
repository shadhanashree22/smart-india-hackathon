## Abstract

The AI-Based Classroom Attendance System is a smart automation solution that leverages face recognition technology to record student attendance in real time.
Traditional manual roll-calling methods are time-consuming and prone to human error. This system overcomes these limitations by using computer vision and machine learning to automatically detect and recognize student faces from a live video feed or uploaded classroom image.

Using the face_recognition and OpenCV libraries, the application encodes each student’s facial features during registration and stores them in a database. During attendance, the system compares the live or uploaded classroom photo with the stored encodings, identifies the students present, and automatically updates attendance logs. A Flask web dashboard allows teachers to view, verify, and export attendance reports.

This project demonstrates how artificial intelligence can streamline classroom management, saving valuable teaching time and ensuring reliable attendance tracking.

## Workflow of the System

The overall workflow of the project can be divided into five main phases, as shown below:

1. Student Registration

Each student’s details — Name, Roll Number, and Face Image — are collected through the web interface.
The system generates and stores a face encoding vector for each student in the local database (attendance.db) and saves the image in the known_faces/ directory.

2. Face Encoding & Storage

The uploaded image is processed using the HOG (Histogram of Oriented Gradients) model via the face_recognition library.
Facial landmarks are identified, and the face is converted into a 128-dimensional encoding vector that uniquely represents that student.

3. Attendance Marking

A classroom photo is uploaded or captured via webcam.
The system detects all faces in the image and compares them with the stored encodings.
For each recognized student, the system automatically marks Present in the attendance table along with the date and time.

4. Database Logging

Attendance data is stored in the SQLite database with the following fields:
Name, Roll Number, Date, Status.
Duplicate entries for the same student on the same day are avoided.

5. Dashboard & Reporting

Teachers can access the Flask dashboard to:
View attendance records.
Verify or correct recognition results.
Export attendance lists to CSV or Excel for reporting.

## app.py

Main Flask application — handles routes, image uploads, and attendance marking.
```
from flask import Flask, render_template, request, redirect, url_for, jsonify
import os
import cv2
import face_recognition
import numpy as np
import sqlite3
from datetime import datetime
from utils import encode_known_faces, mark_attendance

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['KNOWN_FOLDER'] = 'known_faces'

# Create database if not exists
from models import init_db
init_db()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        roll = request.form['roll']
        file = request.files['image']

        if not os.path.exists(app.config['KNOWN_FOLDER']):
            os.makedirs(app.config['KNOWN_FOLDER'])

        path = os.path.join(app.config['KNOWN_FOLDER'], f"{roll}_{name}.jpg")
        file.save(path)

        conn = sqlite3.connect('attendance.db')
        c = conn.cursor()
        c.execute("INSERT INTO students (roll, name, image_path) VALUES (?, ?, ?)", (roll, name, path))
        conn.commit()
        conn.close()

        return render_template('register.html', msg="Student Registered Successfully!")
    return render_template('register.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        file = request.files['photo']
        filename = file.filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        known_encodings, known_names = encode_known_faces(app.config['KNOWN_FOLDER'])
        img = face_recognition.load_image_file(file_path)
        face_locations = face_recognition.face_locations(img)
        face_encodings = face_recognition.face_encodings(img, face_locations)

        recognized = []

        for encodeFace, faceLoc in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(known_encodings, encodeFace)
            faceDist = face_recognition.face_distance(known_encodings, encodeFace)
            matchIndex = np.argmin(faceDist)

            if matches[matchIndex]:
                name = known_names[matchIndex]
                recognized.append(name)
                mark_attendance(name)
        
        return render_template('upload.html', result=recognized)
    return render_template('upload.html')

@app.route('/dashboard')
def dashboard():
    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()
    c.execute("SELECT * FROM attendance")
    data = c.fetchall()
    conn.close()
    return render_template('dashboard.html', data=data)

if __name__ == "__main__":
    app.run(debug=True)
```
## models.py
```
Handles SQLite database schema creation.

import sqlite3

def init_db():
    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()
    
    c.execute('''CREATE TABLE IF NOT EXISTS students (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        roll TEXT UNIQUE,
        name TEXT,
        image_path TEXT
    )''')

    c.execute('''CREATE TABLE IF NOT EXISTS attendance (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        roll TEXT,
        date TEXT,
        status TEXT
    )''')

    conn.commit()
    conn.close()
```
## utils.py

Utility functions for face encoding and attendance marking.
```
import os
import face_recognition
import numpy as np
import sqlite3
from datetime import datetime

def encode_known_faces(known_folder):
    known_encodings = []
    known_names = []
    for filename in os.listdir(known_folder):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            path = os.path.join(known_folder, filename)
            image = face_recognition.load_image_file(path)
            encoding = face_recognition.face_encodings(image)
            if encoding:
                known_encodings.append(encoding[0])
                name = os.path.splitext(filename)[0]
                known_names.append(name)
    return known_encodings, known_names

def mark_attendance(name):
    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()
    roll = name.split("_")[0]
    date_today = datetime.now().strftime("%Y-%m-%d")

    c.execute("SELECT * FROM attendance WHERE roll=? AND date=?", (roll, date_today))
    data = c.fetchone()

    if data is None:
        c.execute("INSERT INTO attendance (name, roll, date, status) VALUES (?, ?, ?, ?)",
                  (name.split("_")[1], roll, date_today, "Present"))
        conn.commit()

    conn.close()
```
🌐 templates/index.html
```
<!DOCTYPE html>
<html>
<head>
    <title>AI Attendance System</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='style.css') }}">
</head>
<body>
    <h1>AI-Based Classroom Attendance</h1>
    <div class="container">
        <a href="{{ url_for('register') }}">Register Student</a>
        <a href="{{ url_for('upload') }}">Upload Attendance Photo</a>
        <a href="{{ url_for('dashboard') }}">View Dashboard</a>
    </div>
</body>
</html>

🧍 templates/register.html
<!DOCTYPE html>
<html>
<head>
    <title>Register Student</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='style.css') }}">
</head>
<body>
    <h2>Register New Student</h2>
    <form method="post" enctype="multipart/form-data">
        <input type="text" name="name" placeholder="Student Name" required><br>
        <input type="text" name="roll" placeholder="Roll Number" required><br>
        <input type="file" name="image" accept="image/*" required><br>
        <button type="submit">Register</button>
    </form>
    {% if msg %}
    <p class="success">{{ msg }}</p>
    {% endif %}
    <a href="/">Back to Home</a>
</body>
</html>
```
## templates/upload.html
```
<!DOCTYPE html>
<html>
<head>
    <title>Mark Attendance</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='style.css') }}">
</head>
<body>
    <h2>Upload Classroom Photo</h2>
    <form method="post" enctype="multipart/form-data">
        <input type="file" name="photo" accept="image/*" required><br>
        <button type="submit">Upload and Mark Attendance</button>
    </form>

    {% if result %}
    <h3>Attendance Marked for:</h3>
    <ul>
        {% for name in result %}
        <li>{{ name }}</li>
        {% endfor %}
    </ul>
    {% endif %}
    <a href="/">Back to Home</a>
</body>
</html>
```
## templates/dashboard.html
```
<!DOCTYPE html>
<html>
<head>
    <title>Attendance Dashboard</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='style.css') }}">
</head>
<body>
    <h2>Attendance Dashboard</h2>
    <table>
        <tr>
            <th>ID</th><th>Name</th><th>Roll</th><th>Date</th><th>Status</th>
        </tr>
        {% for row in data %}
        <tr>
            <td>{{ row[0] }}</td>
            <td>{{ row[1] }}</td>
            <td>{{ row[2] }}</td>
            <td>{{ row[3] }}</td>
            <td>{{ row[4] }}</td>
        </tr>
        {% endfor %}
    </table>
    <a href="/">Back to Home</a>
</body>
</html>
```
## static/style.css

```
body {
    font-family: Arial, sans-serif;
    background-color: #f7f9fc;
    text-align: center;
}

h1, h2 {
    color: #003366;
}

.container a, button {
    display: inline-block;
    margin: 10px;
    padding: 10px 20px;
    background-color: #0055cc;
    color: white;
    text-decoration: none;
    border-radius: 6px;
}

.container a:hover, button:hover {
    background-color: #003d99;
}

table {
    margin: 20px auto;
    border-collapse: collapse;
    width: 80%;
}

table, th, td {
    border: 1px solid #ccc;
    padding: 10px;
}

.success {
    color: green;
    font-weight: bold;
}
```

## requirements.txt
```
Flask
opencv-python
face_recognition
numpy
pandas
```
