import cv2
import numpy as np
import mediapipe as mp
import joblib
import warnings
from collections import deque
import winsound 

# Tắt các cảnh báo không cần thiết
warnings.filterwarnings('ignore')

# --- 1. ĐỊNH NGHĨA LẠI CÁC HÀM VÀ CONSTANTS ---
# (Sử dụng 2-feature, đã xóa iris)

# 1.1. CONSTANTS (Bản đồ Indices)
LEFT_EYE_EAR_INDICES = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_EAR_INDICES = [362, 385, 387, 263, 373, 380]
MOUTH_MAR_INDICES = [61, 82, 13, 312, 291, 317, 14, 87]

# 1.2. CÁC HÀM TÍNH TOÁN
def calculate_ear(eye_points):
    p = [np.array([point.x, point.y]) for point in eye_points]
    vertical_1 = np.linalg.norm(p[1] - p[5])
    vertical_2 = np.linalg.norm(p[2] - p[4])
    numerator = vertical_1 + vertical_2
    horizontal = np.linalg.norm(p[0] - p[3])
    denominator = 2.0 * horizontal
    if denominator == 0: return 0.0
    return numerator / denominator

def calculate_mar(mouth_points):
    p = [np.array([point.x, point.y]) for point in mouth_points]
    vertical_1 = np.linalg.norm(p[1] - p[7])
    vertical_2 = np.linalg.norm(p[2] - p[6])
    vertical_3 = np.linalg.norm(p[3] - p[5])
    numerator = vertical_1 + vertical_2 + vertical_3
    horizontal = np.linalg.norm(p[0] - p[4])
    denominator = 2.0 * horizontal
    if denominator == 0: return 0.0
    return numerator / denominator

# --- 2. KHỞI TẠO CÁC ĐỐI TƯỢNG ---

print("Đang tải mô hình (2-feature) và scaler...")
# 2.1. Tải mô hình và scaler
try:
    model = joblib.load('svm_model.joblib')
    scaler = joblib.load('scaler.joblib')
    print("Tải thành công!")
except FileNotFoundError:
    print("Lỗi: Không tìm thấy file svm_model.joblib hoặc scaler.joblib")
    exit()

# 2.2. Khởi tạo MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=False, 
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# 2.3. Khởi tạo Webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Lỗi: Không thể mở webcam.")
    exit()

# Khởi tạo bộ đệm làm mượt
prediction_buffer = deque(maxlen=25) 

# (THÊM MỚI #2) Biến cờ để kiểm soát báo động
alarm_sounding = False

print("\nĐã mở webcam. Nhấn 'q' để thoát.")

# --- 3. VÒNG LẶP XỬ LÝ VIDEO THỜI GIAN THỰC ---
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    status_text = "KHONG PHAT HIEN KHUON MAT"
    status_color = (0, 0, 255) # Đỏ
    
    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0].landmark
        
        try:
            # --- 3.1. Trích xuất 2 features ---
            left_eye_points = [face_landmarks[i] for i in LEFT_EYE_EAR_INDICES]
            right_eye_points = [face_landmarks[i] for i in RIGHT_EYE_EAR_INDICES]
            avg_ear = (calculate_ear(left_eye_points) + calculate_ear(right_eye_points)) / 2.0

            mouth_points = [face_landmarks[i] for i in MOUTH_MAR_INDICES]
            mar = calculate_mar(mouth_points)

            # --- 3.2. Tạo vector 2-feature và CHUẨN HÓA ---
            features = [avg_ear, mar] 
            features_scaled = scaler.transform([features])

            # (Xóa bớt print debug)
            # print(f"RAW: ear={avg_ear:.2f}, mar={mar:.2f}")
            # print(f"SCALED: {features_scaled[0]}")

            # --- 3.3. DỰ ĐOÁN ---
            prediction = model.predict(features_scaled)
            probability = model.predict_proba(features_scaled)
            label = prediction[0]
            confidence = probability[0][label] * 100

            # --- 3.4. Cập nhật trạng thái (LÀM MƯỢT) ---
            prediction_buffer.append(label)
            try:
                final_label = np.bincount(prediction_buffer).argmax()
            except ValueError:
                final_label = label 

            if final_label == 0:
                status_text = f"ACTIVE ({confidence:.1f}%)"
                status_color = (0, 255, 0) # Xanh lá
                alarm_sounding = False # <-- (THÊM MỚI #3) Reset cờ khi tỉnh
            else:
                status_text = f"FATIGUE ({confidence:.1f}%)"
                status_color = (0, 0, 255) # Đỏ
                
                # Chỉ kêu bíp MỘT LẦN
                if not alarm_sounding:
                    winsound.Beep(1000, 500) # Tần số 1000Hz, kêu 500ms
                    alarm_sounding = True
            
        except Exception as e:
            # print(f"!!! GẶP LỖI: {e}") # (Tắt bớt lỗi)
            status_text = "LOI TRICH XUAT"

    # --- 3.5. Hiển thị kết quả lên frame ---
    cv2.putText(
        frame, 
        status_text, 
        (50, 50), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        1, status_color, 2, cv2.LINE_AA
    )

    cv2.imshow('Drowsiness Detection - Nhan "q" de thoat', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- 4. DỌN DẸP ---
cap.release()
cv2.destroyAllWindows()
face_mesh.close()