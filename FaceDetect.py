import cv2
import os
import numpy as np


class DataCollectorFromFolder:
    def __init__(self, input_folder='dataset'):
        self.input_folder = input_folder
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.hog = cv2.HOGDescriptor()  # Khởi tạo HOG descriptor

    def collect_data(self):
        X = []
        y = []
        for person_name in os.listdir(self.input_folder):
            person_folder = os.path.join(self.input_folder, person_name)
            if os.path.isdir(person_folder):
                for filename in os.listdir(person_folder):
                    img_path = os.path.join(person_folder, filename)
                    print(f"Đang xử lý {img_path}...")  # In đường dẫn file đang xử lý
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

                    if img is None:
                        print(f"Không thể đọc ảnh {filename}")
                        continue

                    # Phát hiện khuôn mặt
                    faces = self.face_cascade.detectMultiScale(img, 1.3, 5)

                    if len(faces) == 0:
                        print(f"Không tìm thấy khuôn mặt trong ảnh {filename}")
                        continue

                    for (x, y_coord, w, h) in faces:
                        face = img[y_coord:y_coord + h, x:x + w]
                        face_resized = cv2.resize(face, (64, 128))  # Resize về kích thước chuẩn 64x128

                        # Trích xuất đặc trưng HOG
                        features = self.extract_features(face_resized)

                        X.append(features)
                        y.append(person_name)
                        print(f"Đã xử lý {filename} của {person_name}")

        return np.array(X), np.array(y)

    def extract_features(self, image):
        # Đảm bảo rằng ảnh có kích thước đúng trước khi trích xuất đặc trưng HOG
        if image.shape != (128, 64):  # Kích thước phải đúng 64x128
            image = cv2.resize(image, (64, 128))
        # Trích xuất đặc trưng HOG
        features = self.hog.compute(image)
        return features.flatten()


# Sử dụng class
collector = DataCollectorFromFolder(input_folder='dataset')

# Tải dataset và trích xuất đặc trưng
X, y = collector.collect_data()
print(f"Đã tải {len(X)} ảnh với {len(set(y))} người khác nhau.")
