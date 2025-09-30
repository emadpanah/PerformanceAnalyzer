<<<<<<< HEAD
# تحلیلگر عملکرد (PerformanceAnalyzer)

## ساختار پروژه
- `videos/`: ویدئوهای ورودی (MP4).
- `output/`: گزارش‌ها و فریم‌های دیباگ.
- `src/`: کدهای منبع.
- `requirements.txt`: وابستگی‌ها.
- `README.md`: توضیحات پروژه.

## نسخه‌های کد
- `src\performanceanalyzer_v1_single_person.py`: تک‌نفره، 3 حالت (برای آموزش).
- `src\performanceanalyzer_v2_multi_person_multi_pose.py`: چندنفره، 4 حالت (نشستن، ایستادن).
- `src\performanceanalyzer_v3_live_camera.py`: live camera + تمام ویژگی‌ها.

## پیش‌نیازها
- Python 3.12
- ویدئوی MP4 (30-60 ثانیه، رزولوشن 640x480 یا 1280x720)

## نصب
1. محیط مجازی:
=======
# performance-analyzer-vision

Pose-based productivity analyzer built with **OpenCV**, **MediaPipe**, **Pandas**, **Streamlit**, and **Plotly**.  
Loads a video, detects pose landmarks, classifies frames into states (Productive / Idle / Away), saves reports, and shows dashboards.

## ✨ Features
- Video ingest (MP4) with optional frame rotation
- Pose landmarks via MediaPipe
- Simple state logic (movement → Productive)
- CSV & JSON reports (duration, FPS, resolution, state distribution)
- Streamlit dashboard: pie + bar charts and a debug frame

## 🧰 Requirements
```txt
opencv-python
mediapipe
numpy
pandas
streamlit
plotly
>>>>>>> 291d73e28bd8e26a21281f030c895c3704491763
