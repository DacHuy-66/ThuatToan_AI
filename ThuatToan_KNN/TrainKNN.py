import cv2
import os
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import joblib


class FaceRecognitionTrainer:
    def __init__(self, dataset_folder='dataset'):
        # Sử dụng đường dẫn tuyệt đối nếu cần
        self.dataset_folder = os.path.abspath(dataset_folder)
        print(f"Đang sử dụng thư mục dataset tại: {self.dataset_folder}")
        self.hog = cv2.HOGDescriptor()  # Khởi tạo HOG descriptor

    def load_dataset(self):
        X = []
        y = []
        # Kiểm tra xem thư mục có tồn tại không
        if not os.path.exists(self.dataset_folder):
            raise FileNotFoundError(f"Không tìm thấy thư mục {self.dataset_folder}")

        for person_name in os.listdir(self.dataset_folder):
            person_folder = os.path.join(self.dataset_folder, person_name)
            if os.path.isdir(person_folder):
                for filename in os.listdir(person_folder):
                    img_path = os.path.join(person_folder, filename)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        face_resized = cv2.resize(img, (64, 128))
                        features = self.extract_features(face_resized)
                        X.append(features)
                        y.append(person_name)
        return np.array(X), np.array(y)

    def extract_features(self, image):
        if image.shape != (64, 128):
            image = cv2.resize(image, (64, 128))
        features = self.hog.compute(image)
        return features.flatten()

    def train_knn(self, X, y):
        print("[INFO] Đang huấn luyện mô hình KNN ...")
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(X, y)
        joblib.dump(knn, '../trainer/knn_model.pkl')
        print("[INFO] Huấn luyện hoàn tất và lưu mô hình KNN.")


if __name__ == "__main__":
    # Thay đổi đường dẫn nếu dataset không ở trong ThuatToan_KNN
    trainer = FaceRecognitionTrainer(dataset_folder='D:/TriTueNhanTao/BTL_AI/dataset')
    X, y = trainer.load_dataset()
    print(f"Đã tải {len(X)} ảnh từ {len(set(y))} người.")
    trainer.train_knn(X, y)

