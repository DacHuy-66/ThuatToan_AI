import cv2
import joblib
import numpy as np
import os


class FaceRecognizerKNN:
    def __init__(self, model_path='../trainer/knn_model.pkl', dataset_path='../dataset'):
        # Load KNN model
        self.knn = joblib.load(model_path)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.hog = cv2.HOGDescriptor()  # Initialize HOG descriptor

        # Load names from dataset folder (thư mục tên người)
        self.names = [person_name for person_name in os.listdir(dataset_path) if
                      os.path.isdir(os.path.join(dataset_path, person_name))]

    def extract_features(self, image):
        # Resize image to 64x128 (consistent with training)
        if image.shape != (128, 64):
            image = cv2.resize(image, (64, 128))
        # Compute HOG features
        features = self.hog.compute(image)
        return features.flatten()

    def recognize_from_camera(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[ERROR] Cannot open camera")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Cannot read frame from camera.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                # Draw rectangle around face
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                face = gray[y:y + h, x:x + w]

                # Extract features
                features = self.extract_features(face)

                # Predict with KNN
                predicted_id = self.knn.predict([features])[0]
                distances, _ = self.knn.kneighbors([features], n_neighbors=1)

                # Calculate confidence based on distance
                distance = distances[0][0]
                max_distance = 100  # Adjust based on your dataset
                confidence = max(0, 100 - (distance / max_distance) * 100)

                # Check if the predicted name is within the known dataset
                if predicted_id in self.names:
                    person_name = predicted_id
                else:
                    person_name = "Unknown"

                # Display name and confidence
                cv2.putText(frame, f"{person_name}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                cv2.putText(frame, f"Confidence: {confidence:.2f}%", (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (255, 255, 0), 2)

            cv2.imshow('Face Recognition', frame)

            # Exit if ESC is pressed
            if cv2.waitKey(1) & 0xFF == 27:
                break

        cap.release()
        cv2.destroyAllWindows()


# Run the recognition
if __name__ == "__main__":
    recognizer = FaceRecognizerKNN(model_path='../trainer/knn_model.pkl', dataset_path='../dataset')
    recognizer.recognize_from_camera()
