import cv2
import face_recognition
import os
from datetime import datetime

known_faces_dir = "Students"
ATTENDANCE_FILE = "Attendance.txt"

known_encodings = []
known_names = []

for filename in os.listdir(known_faces_dir):
    if filename.endswith((".jpg", ".png")):
        image_path = os.path.join(known_faces_dir, filename)
        image = face_recognition.load_image_file(image_path)
        encoding = face_recognition.face_encodings(image)[0]
        known_encodings.append(encoding)
        known_names.append(os.path.splitext(filename)[0])

group_img = face_recognition.load_image_file("attending people.jpg")
group_img_encoding = face_recognition.face_encodings(group_img)
group_img_loc = face_recognition.face_locations(group_img)
group_img_bgr = cv2.cvtColor(group_img, cv2.COLOR_RGB2BGR)

present_people = []

for (face_encoding, (top, right, bottom, left)) in zip(group_img_encoding, group_img_loc):
    match = face_recognition.compare_faces(known_encodings, face_encoding)
    name = "unnknown"
    if True in match:
        match_index = match.index(True)
        name = known_names[match_index]
        if name not in present_people:
            present_people.append(name)
        cv2.rectangle(group_img_bgr, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(group_img_bgr, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

with open(ATTENDANCE_FILE, "a") as f:
    f.write(f"\n--- Attendance on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---\n")
    for person in present_people:
        f.write(person + "\n")

cv2.imwrite("people in class.jpg", group_img_bgr)
cv2.imshow("People that are in class", group_img_bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()



