import warnings
from collections import deque
from typing import Deque, Optional

import av
import joblib
import mediapipe as mp
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw
from streamlit_webrtc import WebRtcMode, VideoProcessorBase, webrtc_streamer


# Suppress non-critical warnings (similar to original script)
warnings.filterwarnings("ignore")


# -----------------------------
# 1. CONSTANTS & GEOMETRY UTILS
# -----------------------------

# Landmark indices (same as in your original run_webcam.py)
LEFT_EYE_EAR_INDICES = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_EAR_INDICES = [362, 385, 387, 263, 373, 380]
MOUTH_MAR_INDICES = [61, 82, 13, 312, 291, 317, 14, 87]


def calculate_ear(eye_points):
    p = [np.array([point.x, point.y]) for point in eye_points]
    vertical_1 = np.linalg.norm(p[1] - p[5])
    vertical_2 = np.linalg.norm(p[2] - p[4])
    numerator = vertical_1 + vertical_2
    horizontal = np.linalg.norm(p[0] - p[3])
    denominator = 2.0 * horizontal
    if denominator == 0:
        return 0.0
    return numerator / denominator


def calculate_mar(mouth_points):
    p = [np.array([point.x, point.y]) for point in mouth_points]
    vertical_1 = np.linalg.norm(p[1] - p[7])
    vertical_2 = np.linalg.norm(p[2] - p[6])
    vertical_3 = np.linalg.norm(p[3] - p[5])
    numerator = vertical_1 + vertical_2 + vertical_3
    horizontal = np.linalg.norm(p[0] - p[4])
    denominator = 2.0 * horizontal
    if denominator == 0:
        return 0.0
    return numerator / denominator


# -----------------------------
# 2. MODEL & SCALER LOADING
# -----------------------------


@st.cache_resource
def load_model_and_scaler():
    """
    Load the SVM model and scaler once and reuse across sessions.
    """
    try:
        model = joblib.load("svm_model.joblib")
        scaler = joblib.load("scaler.joblib")
    except FileNotFoundError:
        st.error("Không tìm thấy file `svm_model.joblib` hoặc `scaler.joblib`. "
                 "Hãy đảm bảo đặt chúng cùng thư mục với `app.py`.")
        st.stop()

    return model, scaler


model, scaler = load_model_and_scaler()


# -----------------------------
# 3. VIDEO PROCESSOR CLASS
# -----------------------------


class DrowsinessVideoProcessor(VideoProcessorBase):
    """
    VideoProcessor for streamlit-webrtc that:
    - Uses MediaPipe Face Mesh to get landmarks
    - Computes EAR (both eyes) and MAR (mouth)
    - Uses your trained SVM + scaler for drowsiness classification
    - Overlays status text on each frame
    """

    def __init__(
        self,
        smoothing_window: int = 25,
        fatigue_prob_threshold: float = 50.0,
    ):
        self.smoothing_window = max(1, int(smoothing_window))
        self.fatigue_prob_threshold = float(fatigue_prob_threshold)

        # Buffer for smoothing predictions
        self.prediction_buffer: Deque[int] = deque(maxlen=self.smoothing_window)

        # MediaPipe Face Mesh initialization
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        # For optional charting
        self.current_ear: Optional[float] = None
        self.current_mar: Optional[float] = None
        self.current_label: Optional[int] = None
        self.current_confidence: Optional[float] = None

        # Internal alarm flag (visual only here; audio not used in web app)
        self.alarm_sounding = False

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        # Get image in BGR format and flip horizontally (mirror effect)
        image_bgr = frame.to_ndarray(format="bgr24")
        image_bgr = np.ascontiguousarray(image_bgr[:, ::-1, :])

        # Convert to RGB for MediaPipe
        rgb_frame = image_bgr[:, :, ::-1]
        results = self.face_mesh.process(rgb_frame)

        status_text = "KHONG PHAT HIEN KHUON MAT"
        status_color = (0, 0, 255)  # Red

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0].landmark

            try:
                # --- Feature extraction (same as original script) ---
                left_eye_points = [face_landmarks[i] for i in LEFT_EYE_EAR_INDICES]
                right_eye_points = [face_landmarks[i] for i in RIGHT_EYE_EAR_INDICES]
                avg_ear = (calculate_ear(left_eye_points) + calculate_ear(right_eye_points)) / 2.0

                mouth_points = [face_landmarks[i] for i in MOUTH_MAR_INDICES]
                mar = calculate_mar(mouth_points)

                self.current_ear = float(avg_ear)
                self.current_mar = float(mar)

                # --- 2-feature vector & scaling ---
                features = [avg_ear, mar]
                features_scaled = scaler.transform([features])

                # --- Prediction ---
                prediction = model.predict(features_scaled)
                probability = model.predict_proba(features_scaled)
                label = int(prediction[0])  # 0 = ACTIVE, 1 = FATIGUE (same convention)

                # Confidence for this label (as percentage)
                confidence = float(probability[0][label] * 100.0)
                self.current_label = label
                self.current_confidence = confidence

                # --- Smoothing ---
                self.prediction_buffer.append(label)
                try:
                    final_label = int(np.bincount(self.prediction_buffer).argmax())
                except ValueError:
                    final_label = label

                # Thresholding by probability if user set > 50%
                is_fatigued = (
                    final_label == 1 and confidence >= self.fatigue_prob_threshold
                )

                if not is_fatigued:
                    status_text = f"ACTIVE ({confidence:.1f}%)"
                    status_color = (0, 255, 0)  # Green
                    self.alarm_sounding = False
                else:
                    status_text = f"FATIGUE ({confidence:.1f}%)"
                    status_color = (0, 0, 255)  # Red
                    # In a browser environment we avoid OS-level beeps
                    self.alarm_sounding = True

            except Exception:
                status_text = "LOI TRICH XUAT"
                status_color = (0, 0, 255)

        # Overlay status text using Pillow on RGB image
        overlay_rgb = rgb_frame.copy()
        pil_img = Image.fromarray(overlay_rgb)
        draw = ImageDraw.Draw(pil_img)

        # Convert BGR color to RGB for Pillow
        text_color_rgb = (status_color[2], status_color[1], status_color[0])
        draw.text((50, 50), status_text, fill=text_color_rgb)

        overlay_rgb = np.array(pil_img)
        # Convert back to BGR for WebRTC output
        output_bgr = overlay_rgb[:, :, ::-1]

        return av.VideoFrame.from_ndarray(output_bgr, format="bgr24")

    def update_params(self, smoothing_window: int, fatigue_prob_threshold: float):
        """
        Update smoothing window and probability threshold from sidebar controls.
        """
        smoothing_window = max(1, int(smoothing_window))
        if smoothing_window != self.smoothing_window:
            self.smoothing_window = smoothing_window
            # Recreate buffer with new size
            self.prediction_buffer = deque(self.prediction_buffer, maxlen=self.smoothing_window)

        self.fatigue_prob_threshold = float(fatigue_prob_threshold)


# -----------------------------
# 4. STREAMLIT UI
# -----------------------------


def main():
    st.set_page_config(
        page_title="Real-time Drowsiness Detection",
        page_icon="😴",
        layout="wide",
    )

    st.title("🚗 Real-time Driver Drowsiness Detection")
    st.markdown(
        """
        Ứng dụng web sử dụng **MediaPipe Face Mesh** và **SVM** để phát hiện buồn ngủ theo thời gian thực.
        
        - Mô hình: SVM (đã huấn luyện trước) dùng 2 đặc trưng: EAR (Eye Aspect Ratio) & MAR (Mouth Aspect Ratio).
        - Video stream được xử lý bằng **streamlit-webrtc** để đảm bảo độ trễ thấp và UI mượt.
        """
    )

    # ---- Sidebar controls ----
    st.sidebar.header("⚙️ Cài đặt")

    smoothing_window = st.sidebar.slider(
        "Độ mượt dự đoán (số khung hình)",
        min_value=5,
        max_value=50,
        value=25,
        step=1,
        help="Sử dụng trung bình số khung hình này để làm mượt dự đoán (giảm nhiễu).",
    )

    fatigue_prob_threshold = st.sidebar.slider(
        "Ngưỡng xác suất buồn ngủ (%)",
        min_value=50.0,
        max_value=99.0,
        value=70.0,
        step=1.0,
        help="Nếu xác suất mô hình > ngưỡng này và nhãn là FATIGUE thì coi là buồn ngủ.",
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Trạng thái hiện tại** sẽ được hiển thị trên video.")

    # ---- Main content layout ----
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📷 Webcam")

        # WebRTC configuration for public deployment (STUN server)
        rtc_configuration = {
            "iceServers": [
                {"urls": ["stun:stun.l.google.com:19302"]},
            ]
        }

        webrtc_ctx = webrtc_streamer(
            key="drowsiness-detection",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=rtc_configuration,
            media_stream_constraints={"video": True, "audio": False},
            video_processor_factory=lambda: DrowsinessVideoProcessor(
                smoothing_window=smoothing_window,
                fatigue_prob_threshold=fatigue_prob_threshold,
            ),
        )

    with col2:
        st.subheader("📊 Thông số thời gian thực")
        ear_placeholder = st.metric("EAR (Eye Aspect Ratio)", value="-")
        mar_placeholder = st.metric("MAR (Mouth Aspect Ratio)", value="-")
        status_placeholder = st.empty()

        ear_chart = st.empty()

        # Store history in session_state for charting
        if "ear_history" not in st.session_state:
            st.session_state["ear_history"] = []

        # Real-time stats / chart loop (polling)
        # Note: Streamlit reruns the script automatically; no infinite loop here.
        if webrtc_ctx and webrtc_ctx.video_processor:
            processor: DrowsinessVideoProcessor = webrtc_ctx.video_processor

            # Sync sidebar parameters with processor (in case user changed them)
            processor.update_params(
                smoothing_window=smoothing_window,
                fatigue_prob_threshold=fatigue_prob_threshold,
            )

            if processor.current_ear is not None:
                ear_placeholder.metric(
                    "EAR (Eye Aspect Ratio)", f"{processor.current_ear:.3f}"
                )
                st.session_state["ear_history"].append(processor.current_ear)

            if processor.current_mar is not None:
                mar_placeholder.metric(
                    "MAR (Mouth Aspect Ratio)", f"{processor.current_mar:.3f}"
                )

            # Status text
            if processor.current_label is not None and processor.current_confidence is not None:
                if processor.current_label == 0:
                    status_placeholder.markdown(
                        "<span style='color:limegreen; font-size:24px; font-weight:bold;'>✅ ACTIVE</span>",
                        unsafe_allow_html=True,
                    )
                else:
                    status_placeholder.markdown(
                        "<span style='color:red; font-size:24px; font-weight:bold;'>⚠️ DROWSY!</span>",
                        unsafe_allow_html=True,
                    )

            # Real-time EAR chart (optional)
            if len(st.session_state["ear_history"]) > 1:
                ear_chart.line_chart(st.session_state["ear_history"])
        else:
            status_placeholder.info(
                "Bật webcam ở phần bên trái để xem thông số và trạng thái."
            )


if __name__ == "__main__":
    main()


