import cv2
import pickle
import numpy as np
import os

# Load model KNN đã huấn luyện
with open('trainer/knn_model.pkl', 'rb') as f:
    knn = pickle.load(f)

# Khởi tạo bộ phát hiện khuôn mặt
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

# Khởi tạo font
font = cv2.FONT_HERSHEY_DUPLEX

# Danh sách các tên tương ứng với ID
names = ['0', 'DacHuy', 'Anh', 'Ly', '4', '5', '6', '7', '8', '9', '10']

# Khởi tạo camera
cam = cv2.VideoCapture(0)
if not cam.isOpened():
    print("[ERROR] Không thể mở camera")
    exit()

# Thiết lập độ phân giải của video
cam.set(3, 420)  # Chiều rộng
cam.set(4, 340)  # Chiều cao

# Thay đổi kích thước vùng khuôn mặt nhỏ nhất
minW = 0.1 * cam.get(3)
minH = 0.1 * cam.get(4)

while True:
    ret, img = cam.read()
    if not ret:
        print("[ERROR] Không thể đọc được hình ảnh từ camera")
        break

    # Lật hình ảnh theo chiều ngang
    img = cv2.flip(img, 1)

    # Chuyển sang ảnh xám
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Phát hiện khuôn mặt
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=7, minSize=(int(minW), int(minH)))

    for (x, y, w, h) in faces:
        # Vẽ hình chữ nhật quanh khuôn mặt
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Cắt khuôn mặt và resize về 100x100
        face = gray[y:y + h, x:x + w]
        face_resized = cv2.resize(face, (100, 100))  # Resize về kích thước 100x100
        face_flatten = face_resized.flatten().reshape(1, -1)  # Flatten và reshape về đúng kích thước (1, 10000)

        # Nhận diện khuôn mặt
        id = knn.predict(face_flatten)[0]
        confidence = knn.predict_proba(face_flatten).max() * 100

        # Hiển thị ID và độ tin cậy
        if confidence > 75:  # Điều chỉnh ngưỡng độ tin cậy
            name = names[id]
        else:
            name = "unknown"

        confidence_text = f"{confidence:.2f}%"
        cv2.putText(img, str(name), (x + 5, y - 30), font, 1, (255, 255, 255), 2)
        cv2.putText(img, confidence_text, (x + 5, y - 5), font, 1, (255, 255, 0), 1)

    # Hiển thị hình ảnh với khuôn mặt được nhận diện
    cv2.imshow('Nhan dien khuon mat', img)

    # Kiểm tra phím nhấn (ESC để thoát)
    k = cv2.waitKey(10) & 0xff
    if k == 27:
        break

print("\n [INFO] Thoát")
cam.release()
cv2.destroyAllWindows()
