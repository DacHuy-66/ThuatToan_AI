import cv2
import os
import time
import numpy as np


class CameraDataCollector:
    def __init__(self, camera_id=0, output_folder='dataset'):
        self.camera_id = camera_id
        self.output_folder = output_folder
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

    def start_capture(self, person_name, num_images=20, delay=1):
        cap = cv2.VideoCapture(self.camera_id)

        if not cap.isOpened():
            print("Không thể mở camera.")
            return

        person_folder = os.path.join(self.output_folder, person_name)
        if not os.path.exists(person_folder):
            os.makedirs(person_folder)

        count = 0
        while count < num_images:
            ret, frame = cap.read()
            if not ret:
                print("Không thể đọc frame từ camera.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                face = gray[y:y + h, x:x + w]
                face_resized = cv2.resize(face, (100, 100))

                filename = f"{person_name}_{count}.jpg"
                cv2.imwrite(os.path.join(person_folder, filename), face_resized)
                count += 1
                print(f"Đã lưu {filename}")

            cv2.imshow('frame', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            time.sleep(delay)

        cap.release()
        cv2.destroyAllWindows()
        print(f"Đã thu thập {count} ảnh cho {person_name}")

    def collect_data(self):
        while True:
            person_name = input("Nhập tên người (hoặc 'q' để thoát): ")
            if person_name.lower() == 'q':
                break

            num_images = int(input("Số lượng ảnh cần thu thập: "))
            self.start_capture(person_name, num_images)

    @staticmethod
    def preprocess_image(image):
        return cv2.resize(image, (100, 100))

    @staticmethod
    def extract_features(image):
        # Sử dụng HOG để trích xuất đặc trưng
        hog = cv2.HOGDescriptor()
        features = hog.compute(image)
        return features.flatten()

    def load_dataset(self):
        X = []
        y = []
        for person_name in os.listdir(self.output_folder):
            person_folder = os.path.join(self.output_folder, person_name)
            if os.path.isdir(person_folder):
                for filename in os.listdir(person_folder):
                    img_path = os.path.join(person_folder, filename)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        preprocessed_img = self.preprocess_image(img)
                        features = self.extract_features(preprocessed_img)
                        X.append(features)
                        y.append(person_name)
        return np.array(X), np.array(y)


# Sử dụng class
if __name__ == "__main__":
    collector = CameraDataCollector()
    collector.collect_data()

    # Tải dataset và trích xuất đặc trưng
    X, y = collector.load_dataset()
    print(f"Đã tải {len(X)} ảnh với {len(set(y))} người khác nhau.")