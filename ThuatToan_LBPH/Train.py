import cv2
import numpy as np
from PIL import Image
import os

# Đường dẫn đến thư mục dataset
path = '../dataset'

# Khởi tạo LBPHFaceRecognizer
recognizer = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8)

# Khởi tạo bộ phát hiện khuôn mặt
detector = cv2.CascadeClassifier("../haarcascade_frontalface_default.xml")


def getImagesAndLabels(path):
    faceSamples = []
    ids = []

    # Duyệt qua từng thư mục con (tên thư mục là tên người)
    for person_name in os.listdir(path):
        person_folder = os.path.join(path, person_name)
        if os.path.isdir(person_folder):
            # Duyệt qua các ảnh trong mỗi thư mục con
            for filename in os.listdir(person_folder):
                imagePath = os.path.join(person_folder, filename)
                try:
                    # Mở ảnh và chuyển sang ảnh xám
                    PIL_img = Image.open(imagePath).convert('L')
                    img_numpy = np.array(PIL_img, 'uint8')

                    # Phát hiện khuôn mặt
                    faces = detector.detectMultiScale(img_numpy)

                    # Lưu khuôn mặt và ID (ID là tên thư mục, person_name)
                    for (x, y, w, h) in faces:
                        faceSamples.append(img_numpy[y:y + h, x:x + w])
                        ids.append(person_name)  # Lưu tên người thay vì ID số
                except Exception as e:
                    print(f"Không thể xử lý {imagePath}: {e}")
    return faceSamples, ids


print("\n[INFO] Đang training dữ liệu ...")
faces, ids = getImagesAndLabels(path)

# Chuyển tên thành các số ID để tương thích với LBPH
unique_names = list(set(ids))
name_to_id = {name: idx for idx, name in enumerate(unique_names)}
ids_numeric = [name_to_id[name] for name in ids]

# Huấn luyện mô hình nhận diện khuôn mặt
recognizer.train(faces, np.array(ids_numeric))

# Lưu mô hình đã huấn luyện
recognizer.write('../trainer/trainer.yml')

# Thông báo hoàn tất và số lượng người đã train
print(f"\n[INFO] Đã train {len(unique_names)} người. Thoát.")
