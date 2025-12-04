# 🚗 Real-Time Driver Drowsiness Detection

Real-time drowsiness detection web app using **MediaPipe Face Mesh** and **SVM** classifier. Detects driver fatigue through Eye Aspect Ratio (EAR) and Mouth Aspect Ratio (MAR) analysis.

## Features

- ✅ Real-time webcam processing via WebRTC
- ✅ SVM-based drowsiness classification
- ✅ Visual and audio alerts
- ✅ Configurable sensitivity controls
- ✅ Live metrics dashboard

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Run Locally

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser and allow camera access.

## Deployment

Deploy to **Streamlit Cloud**:

1. Push repository to GitHub
2. Connect to [Streamlit Cloud](https://streamlit.io/cloud)
3. Point to `app.py` as entrypoint
4. Deploy

**Required files:**
- `app.py`
- `requirements.txt`
- `svm_model.joblib`
- `scaler.joblib`

## Usage

1. **Enable webcam** - Click "Start" in the video panel
2. **Adjust settings** (sidebar):
   - Smoothing window: prediction stability (5-50 frames)
   - Fatigue threshold: detection sensitivity (50-99%)
   - Sound alert: toggle audio warnings
3. **Monitor status**:
   - 🟢 **ACTIVE**: Driver is alert
   - 🔴 **FATIGUE**: Drowsiness detected (triggers sound alert)

## Technical Details

- **Features**: EAR (Eye Aspect Ratio) + MAR (Mouth Aspect Ratio)
- **Model**: Pre-trained SVM classifier (`svm_model.joblib`)
- **Preprocessing**: StandardScaler normalization (`scaler.joblib`)
- **Landmarks**: MediaPipe Face Mesh (468 points)
- **Streaming**: streamlit-webrtc for low-latency video processing

## Requirements

See `requirements.txt` for full list. Key dependencies:
- `streamlit`, `streamlit-webrtc`
- `mediapipe`, `opencv-python-headless`
- `scikit-learn`, `joblib`
- `numpy`, `Pillow`


