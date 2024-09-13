import cv2
import os
import time

# Khởi tạo camera
cam = cv2.VideoCapture(0)

# Kiểm tra xem camera có hoạt động không
if not cam.isOpened():
    print("[ERROR] Không thể mở camera")
    exit()

# Thiết lập độ phân giải
cam.set(3, 320)  # Chiều rộng giảm để tăng tốc độ xử lý
cam.set(4, 240)  # Chiều cao

# Khởi tạo bộ phát hiện khuôn mặt
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Nhập ID khuôn mặt
face_id = input('\nNhập ID khuôn mặt <return> ==> ')

# Thông báo khởi tạo
print("\n[INFO] Khởi tạo camera ...")

# Biến đếm số ảnh đã chụp
count = 0

while True:
    ret, img = cam.read()

    if not ret:
        print("[ERROR] Không thể đọc được hình ảnh từ camera")
        break

    # Lật ảnh để phản chiếu theo chiều dọc (tùy chọn)
    img = cv2.flip(img, 1)  # flip video image horizontally

    # Chuyển ảnh sang màu xám để phát hiện khuôn mặt
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Phát hiện khuôn mặt
    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,  # Giảm để tăng độ chính xác
        minNeighbors=7,  # Tăng để lọc bỏ các phát hiện sai
        minSize=(40, 40),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Vẽ hình chữ nhật và lưu khuôn mặt vào dataset
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        count += 1

        # Resize khuôn mặt về 100x100 trước khi lưu
        face_resized = cv2.resize(gray[y:y + h, x:x + w], (100, 100))
        cv2.imwrite(f"dataset/User.{face_id}.{count}.jpg", face_resized)

        # Hiển thị số lượng ảnh đã chụp lên hình ảnh
        cv2.putText(img, f'So luong anh: {count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Hiển thị hình ảnh với khung khuôn mặt
        cv2.imshow('image', img)

        # Thêm thời gian nghỉ giữa mỗi lần chụp ảnh (giúp giảm tải camera và tránh chụp quá nhanh)
        time.sleep(2)  # Thay đổi thời gian nghỉ giữa mỗi lần chụp ảnh thành 2 giây

    # Kiểm tra phím nhấn (ESC để thoát)
    k = cv2.waitKey(100) & 0xff
    if k == 27:  # Nhấn ESC để thoát
        break
    elif count >= 8:  # Dừng lại sau khi đã chụp đủ 8 ảnh
        break

# Thông báo kết thúc
print("\n[INFO] Thoát")

# Giải phóng camera và đóng các cửa sổ
cam.release()
cv2.destroyAllWindows()
