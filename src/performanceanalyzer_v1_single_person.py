import cv2  # برای پردازش ویدئو و تصاویر
import mediapipe as mp  # برای تشخیص حالت بدن
import numpy as np  # برای محاسبات عددی
import pandas as pd  # برای مدیریت داده‌ها
import streamlit as st  # برای داشبورد
import plotly.express as px  # برای نمودارها
import argparse  # برای آرگومان‌های خط فرمان
import os  # برای مدیریت فایل
import time  # برای ذخیره فریم‌های دیباگ

# تنظیم MediaPipe برای تشخیص حالت بدن
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.2)
mp_drawing = mp.solutions.drawing_utils

# تابع اصلاح چرخش ویدئو
def rotate_frame(frame, rotation_code=cv2.ROTATE_90_COUNTERCLOCKWISE):
    return cv2.rotate(frame, rotation_code)

# تابع تشخیص حالت در هر فریم
def detect_state(frame, prev_landmarks):
    state = "Away/Moving"
    movement_detected = False
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)
    
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        cv2.imshow('Debug Frame', frame)
        cv2.waitKey(1)
    
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        if landmarks[mp_pose.PoseLandmark.LEFT_HIP].visibility > 0.3 and \
           landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].visibility > 0.3:
            if prev_landmarks:
                dist = np.linalg.norm(np.array([landmarks[mp_pose.PoseLandmark.NOSE].x,
                                               landmarks[mp_pose.PoseLandmark.NOSE].y]) -
                                     np.array([prev_landmarks[mp_pose.PoseLandmark.NOSE].x,
                                               prev_landmarks[mp_pose.PoseLandmark.NOSE].y]))
                if dist > 0.05:
                    movement_detected = True
            
            state = "Productive" if movement_detected else "Idle"
        
        return state, results.pose_landmarks.landmark, frame  # تغییر به .landmark
    return state, None, frame

# تابع تحلیل ویدئو
def analyze_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("خطا: نمی‌توان فایل ویدئو را باز کرد")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps
    
    states = []
    prev_landmarks = None
    frame_count = 0
    debug_frames = []
    
    output_dir = "output"
    frames_dir = os.path.join(output_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = rotate_frame(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)  # اصلاح چرخش
        frame_count += 1
        if frame_count % 3 != 0:
            continue
        state, prev_landmarks, debug_frame = detect_state(frame, prev_landmarks)
        states.append(state)
        
        if frame_count % 10 == 0:
            cv2.imwrite(os.path.join(frames_dir, f"frame_{frame_count}.jpg"), debug_frame)
            debug_frames.append(debug_frame)
    
    cap.release()
    cv2.destroyAllWindows()
    
    state_counts = pd.Series(states).value_counts(normalize=True) * 100
    productive_pct = state_counts.get("Productive", 0)
    unproductive_pct = 100 - productive_pct
    
    report = {
        "مدت زمان کل (ثانیه)": duration,
        "رزولوشن": f"{width}x{height}",
        "فریم در ثانیه": fps,
        "تعداد فریم‌ها": total_frames,
        "درصد مولد": productive_pct,
        "درصد غیرمولد": unproductive_pct,
        "تفکیک حالات": state_counts.to_dict()
    }
    
    pd.DataFrame([report]).to_csv(os.path.join(output_dir, "analysis_report_v1.csv"), index=False)
    return report, state_counts, debug_frames

# تابع داشبورد
def show_dashboard(report=None, state_counts=None, debug_frames=None):
    st.title("📊 داشبورد تحلیلگر عملکرد (تک‌نفره)")
    
    if report is None:
        st.header("آپلود ویدئو")
        video_file = st.file_uploader("ویدئو را انتخاب کنید (MP4)", type=["mp4"])
        if video_file:
            video_path = os.path.join("videos", "uploaded_video.mp4")
            os.makedirs("videos", exist_ok=True)
            with open(video_path, "wb") as f:
                f.write(video_file.getbuffer())
            st.success("ویدئو آپلود شد. تحلیل در حال انجام...")
            report, state_counts, debug_frames = analyze_video(video_path)
    
    if report:
        st.header("اطلاعات ویدئو")
        st.write(f"مدت زمان: {report['مدت زمان کل (ثانیه)']} ثانیه")
        st.write(f"رزولوشن: {report['رزولوشن']}")
        st.write(f"فریم در ثانیه: {report['فریم در ثانیه']}")
        st.write(f"تعداد فریم‌ها: {report['تعداد فریم‌ها']}")
        
        st.header("نتایج تحلیل")
        fig = px.pie(values=[report["درصد مولد"], report["درصد غیرمولد"]],
                     names=["مولد", "غیرمولد"],
                     title="تفکیک بهره‌وری",
                     color_discrete_sequence=["#00CC96", "#EF553B"])
        st.plotly_chart(fig)
        
        fig_states = px.bar(x=state_counts.index, y=state_counts.values,
                            labels={"x": "حالت", "y": "درصد (%)"},
                            title="توزیع حالات",
                            color=state_counts.index,
                            color_discrete_map={"Productive": "#00CC96", "Idle": "#EF553B", "Away/Moving": "#636EFA"})
        st.plotly_chart(fig_states)
        
        st.header("گزارش کامل")
        st.table(pd.DataFrame([report]))
        
        st.header("فریم نمونه (دیباگ)")
        if debug_frames:
            frame = debug_frames[0]
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            st.image(frame_rgb, caption="فریم نمونه با نقاط کلیدی", use_column_width=True)
        
        st.header("ذخیره گزارش")
        if st.button("ذخیره گزارش به JSON"):
            pd.DataFrame([report]).to_json(os.path.join("output", "analysis_report_v1.json"), orient="records", lines=True, force_ascii=False)
            st.success("گزارش به JSON ذخیره شد")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="تحلیلگر عملکرد (تک‌نفره)")
    parser.add_argument("--video", type=str, help="مسیر فایل ویدئو")
    parser.add_argument("--dashboard", action="store_true", help="نمایش داشبورد پس از تحلیل")
    args = parser.parse_args()
    
    if args.video:
        report, state_counts, debug_frames = analyze_video(args.video)
        print("تحلیل کامل شد. گزارش در output/analysis_report_v1.csv ذخیره شد")
        print(report)
        if args.dashboard:
            show_dashboard(report, state_counts, debug_frames)
    else:
        show_dashboard()