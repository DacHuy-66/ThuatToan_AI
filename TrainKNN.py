import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import os
from PIL import Image

path = 'dataset'

# Khởi tạo bộ phát hiện khuôn mặt
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Tạo danh sách để lưu đặc trưng khuôn mặt và nhãn
faceSamples = []
ids = []

# Hàm trích xuất ảnh và nhãn từ dataset
def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    for imagePath in imagePaths:
        PIL_img = Image.open(imagePath).convert('L')  # Chuyển sang ảnh xám
        img_numpy = np.array(PIL_img, 'uint8')

        # Lấy ID từ tên file
        id = int(os.path.split(imagePath)[-1].split(".")[1])

        # Phát hiện khuôn mặt trong ảnh
        faces = detector.detectMultiScale(img_numpy)

        for (x, y, w, h) in faces:
            # Resize khuôn mặt về kích thước cố định (ví dụ: 100x100)
            face_resized = cv2.resize(img_numpy[y:y + h, x:x + w], (100, 100))
            faceSamples.append(face_resized.flatten())  # Lưu đặc trưng mặt
            ids.append(id)

    return faceSamples, ids

print("\n [INFO] Đang huấn luyện với KNN ...")
faces, ids = getImagesAndLabels(path)

# Tạo model KNN
knn = KNeighborsClassifier(n_neighbors=5)

# Huấn luyện KNN
knn.fit(faces, ids)

# Lưu model KNN
import pickle
with open('trainer/knn_model.pkl', 'wb') as f:
    pickle.dump(knn, f)

print("\n [INFO] Huấn luyện hoàn tất.")
