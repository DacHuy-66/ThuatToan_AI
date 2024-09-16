import cv2
import numpy as np
import os

# Khởi tạo bộ nhận diện khuôn mặt
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('../trainer/trainer.yml')

# Khởi tạo bộ phát hiện khuôn mặt
cascadePath = "../haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

# Khởi tạo font
font = cv2.FONT_HERSHEY_DUPLEX

# Tự động đọc danh sách tên từ thư mục dataset
dataset_path = '../dataset'
names = [person_name for person_name in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, person_name))]

# Khởi tạo camera
cam = cv2.VideoCapture(0)
if not cam.isOpened():
    print("[ERROR] Không thể mở camera")
    exit()

# Thiết lập độ phân giải của video
cam.set(3, 420)  # Chiều rộng (giảm để tăng tốc độ)
cam.set(4, 340)  # Chiều cao (giảm để tăng tốc độ)

# Thay đổi kích thước vùng khuôn mặt nhỏ nhất
minW = 0.1 * cam.get(3)
minH = 0.1 * cam.get(4)

while True:
    ret, img = cam.read()
    if not ret:
        print("[ERROR] Không thể đọc được hình ảnh từ camera")
        break

    # Điều chỉnh lật hình ảnh theo yêu cầu
    img = cv2.flip(img, 1)

    # Chuyển hình ảnh sang màu xám để phát hiện khuôn mặt
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Phát hiện khuôn mặt
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=7, minSize=(int(minW), int(minH)))

    for (x, y, w, h) in faces:
        # Vẽ hình chữ nhật quanh khuôn mặt
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Nhận diện khuôn mặt
        id, confidence = recognizer.predict(gray[y:y + h, x:x + w])

        if confidence < 87:
            if id < len(names):
                person_name = names[id]
            else:
                person_name = "Unknown"
            confidence_text = " {0}%".format(round(100 - confidence))
        else:
            person_name = "Unknown"
            confidence_text = " {0}%".format(round(100 - confidence))

        # Hiển thị ID và độ tin cậy
        cv2.putText(img, str(person_name), (x + 5, y - 30), font, 1, (255, 255, 255), 2)
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
