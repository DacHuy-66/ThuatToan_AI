import cv2
import face_recognition
import numpy as np
import pickle
import os

# Thư mục lưu hình ảnh sinh viên
dataset_path = 'D:/TriTueNhanTao/BTL_AI/Data'
encodings_file = './encodings.pickle'

# Tạo một danh sách để lưu embeddings và tên sinh viên
known_face_encodings = []
known_face_names = []

# Duyệt qua từng file hình ảnh trong thư mục dataset
for student_name in os.listdir(dataset_path):
    image_path = os.path.join(dataset_path, student_name)
    image = cv2.imread(image_path)

    # Chuyển ảnh từ BGR sang RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Phát hiện vị trí khuôn mặt
    boxes = face_recognition.face_locations(rgb_image, model='hog')

    # Trích xuất đặc trưng khuôn mặt
    encodings = face_recognition.face_encodings(rgb_image, boxes)

    # Giả sử mỗi hình ảnh chỉ có một khuôn mặt
    if len(encodings) > 0:
        known_face_encodings.append(encodings[0])
        known_face_names.append(student_name.split('.')[0])

# Lưu các embeddings và tên sinh viên vào file
data = {"encodings": known_face_encodings, "names": known_face_names}
with open(encodings_file, "wb") as f:
    pickle.dump(data, f)

print("Đã lưu trữ các đặc trưng khuôn mặt vào file.")
