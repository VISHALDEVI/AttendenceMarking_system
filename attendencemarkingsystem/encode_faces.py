import face_recognition
import os
import pickle

IMAGE_DIR = "image"
encodings = []
names = []

for filename in os.listdir(IMAGE_DIR):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        path = os.path.join(IMAGE_DIR, filename)
        image = face_recognition.load_image_file(path)
        face_encs = face_recognition.face_encodings(image)
        if face_encs:
            encodings.append(face_encs[0])
            names.append(os.path.splitext(filename)[0])
            print(f"Encoded: {filename}")
        else:
            print(f"No face found in: {filename}")

with open("encodings.pkl", "wb") as f:
    pickle.dump({"encodings": encodings, "names": names}, f)

print(f"\nDone! {len(encodings)} face(s) encoded.")
