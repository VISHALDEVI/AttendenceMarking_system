import cv2
import face_recognition
import os
import csv
from datetime import datetime

IMAGE_DIR = "image"
CSV_FILE = "attendance.csv"

# Load all known faces from image folder
known_encodings = []
known_names = []

for filename in os.listdir(IMAGE_DIR):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        img = face_recognition.load_image_file(os.path.join(IMAGE_DIR, filename))
        encs = face_recognition.face_encodings(img)
        if encs:
            known_encodings.append(encs[0])
            known_names.append(os.path.splitext(filename)[0])
            print(f"Loaded: {filename}")

if not known_encodings:
    print("No faces found in image folder.")
    exit()

# Write CSV header if file doesn't exist
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, "w", newline="") as f:
        csv.writer(f).writerow(["Name", "Date", "Time"])

# Check who is already marked today
def already_marked_today(name):
    today = datetime.now().strftime("%Y-%m-%d")
    if not os.path.exists(CSV_FILE):
        return False
    with open(CSV_FILE, "r") as f:
        for row in csv.reader(f):
            if len(row) >= 2 and row[0] == name and row[1] == today:
                return True
    return False

# Track who has been marked this session
marked_today = set(name for name in known_names if already_marked_today(name))

if marked_today == set(known_names):
    print("Attendance already marked today for all people.")
    exit()

cap = cv2.VideoCapture(0)
print("Press Q to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

    locations = face_recognition.face_locations(rgb)
    encodings = face_recognition.face_encodings(rgb, locations)

    for enc, loc in zip(encodings, locations):
        matches = face_recognition.compare_faces(known_encodings, enc, tolerance=0.5)
        name = "Unknown"

        if True in matches:
            name = known_names[matches.index(True)]

        # Mark attendance if not already marked today
        if name != "Unknown" and name not in marked_today:
            marked_today.add(name)
            now = datetime.now()
            with open(CSV_FILE, "a", newline="") as f:
                csv.writer(f).writerow([name, now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S")])
            print(f"Attendance marked: {name}")

        # Draw box and label
        top, right, bottom, left = [v * 4 for v in loc]
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Attendance System", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
