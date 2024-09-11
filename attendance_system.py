import cv2
import face_recognition
import numpy as np
import pickle
from datetime import datetime

# Tải các embeddings và tên sinh viên từ file
encodings_file = './encodings.pickle'
with open(encodings_file, "rb") as f:
    data = pickle.load(f)

known_face_encodings = data["encodings"]
known_face_names = data["names"]

# Danh sách sinh viên đã điểm danh
attendance_list = []


def mark_attendance(name):
    """ Ghi lại tên sinh viên và thời gian điểm danh """
    with open('attendance.csv', 'a') as f:
        now = datetime.now()
        dt_string = now.strftime('%Y-%m-%d %H:%M:%S')
        f.write(f'{name},{dt_string}\n')
    print(f'{name} đã điểm danh lúc {dt_string}')


# Mở webcam
video_capture = cv2.VideoCapture(0)

while True:
    # Lấy từng frame từ video
    ret, frame = video_capture.read()

    # Chuyển ảnh từ BGR sang RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Phát hiện vị trí khuôn mặt trong frame
    face_locations = face_recognition.face_locations(rgb_frame, model='hog')

    # Trích xuất đặc trưng khuôn mặt từ frame
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Duyệt qua từng khuôn mặt được phát hiện
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):

        # So sánh khuôn mặt mới với các khuôn mặt đã biết
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        # Nếu có sự trùng khớp
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)

        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        # Hiển thị tên sinh viên lên frame
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        # Nếu sinh viên chưa được điểm danh, ghi lại thông tin
        if name != "Unknown" and name not in attendance_list:
            mark_attendance(name)
            attendance_list.append(name)

    # Hiển thị frame
    cv2.imshow('Video', frame)

    # Bấm 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Dừng webcam
video_capture.release()
cv2.destroyAllWindows()
