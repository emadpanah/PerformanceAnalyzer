import cv2  # برای پردازش ویدئو و تصاویر
import mediapipe as mp  # برای تشخیص حالت بدن
import numpy as np  # برای محاسبات عددی
import pandas as pd  # برای مدیریت داده‌ها
import streamlit as st  # برای داشبورد
import plotly.express as px  # برای نمودارها
import argparse  # برای آرگومان‌های خط فرمان
import os  # برای مدیریت فایل
import time  # برای ذخیره فریم‌های دیباگ
from ultralytics import YOLO  # برای تشخیص افراد

# تنظیم MediaPipe برای تشخیص حالت بدن (تک‌نفره روی crop)
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.05, model_complexity=2, min_tracking_confidence=0.1)
mp_drawing = mp.solutions.drawing_utils

# مدل YOLO برای تشخیص افراد
yolo_model = YOLO('yolov8n.pt')  # مدل سبک YOLOv8

# تابع اصلاح چرخش ویدئو
def rotate_frame(frame, rotation_code=cv2.ROTATE_90_COUNTERCLOCKWISE):
    if rotation_code is None:
        return frame
    return cv2.rotate(frame, rotation_code)

# تابع تشخیص حالت برای هر فرد
def detect_state(landmarks, prev_landmarks, frame_height):
    state = "Away"
    movement_detected = False
    
    if landmarks:
        hip_y = landmarks[mp_pose.PoseLandmark.LEFT_HIP].y
        shoulder_y = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y
        if landmarks[mp_pose.PoseLandmark.LEFT_HIP].visibility > 0.2 and \
           landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].visibility > 0.2:
            # تشخیص نشستن یا ایستادن
            if hip_y > 0.4 * frame_height:  # کاهش آستانه برای ویدئوهای عمودی
                state = "Sitting Idle"
            else:
                state = "Standing Idle"
            
            if prev_landmarks:
                dist = np.linalg.norm(np.array([landmarks[mp_pose.PoseLandmark.NOSE].x,
                                               landmarks[mp_pose.PoseLandmark.NOSE].y]) -
                                     np.array([prev_landmarks[mp_pose.PoseLandmark.NOSE].x,
                                               prev_landmarks[mp_pose.PoseLandmark.NOSE].y]))
                if dist > 0.02 and state == "Standing Idle":  # کاهش آستانه حرکت
                    state = "Standing Moving"
                elif dist > 0.02:
                    state = "Productive"
    
    return state

# تابع تحلیل ویدئو (چندنفره با YOLO + MediaPipe)
def analyze_video(video_path, rotation_code=cv2.ROTATE_90_COUNTERCLOCKWISE):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("خطا: نمی‌توان فایل ویدئو را باز کرد")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps
    
    states_history = [[] for _ in range(4)]  # حداکثر 4 نفر
    prev_landmarks_list = [None for _ in range(4)]
    frame_count = 0
    debug_frames = []
    detection_log = []
    
    output_dir = "output"
    frames_dir = os.path.join(output_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = rotate_frame(frame, rotation_code)
        frame_count += 1
        if frame_count % 3 != 0:
            continue
        
        # تشخیص افراد با YOLO
        yolo_results = yolo_model(frame)
        num_persons = 0
        for result in yolo_results:
            boxes = result.boxes
            for box in boxes:
                if num_persons >= 4:
                    break
                if box.cls == 0:  # class 0 = person
                    x1, y1, x2, y2 = box.xyxy[0].int().tolist()
                    crop = frame[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue
                    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    pose_results = pose.process(crop_rgb)
                    
                    if pose_results.pose_landmarks:
                        state = detect_state(pose_results.pose_landmarks.landmark, prev_landmarks_list[num_persons], height)
                        states_history[num_persons].append(state)
                        prev_landmarks_list[num_persons] = pose_results.pose_landmarks.landmark
                        mp_drawing.draw_landmarks(crop, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                        frame[y1:y2, x1:x2] = crop
                        num_persons += 1
        
        print(f"فریم {frame_count}: {num_persons} نفر تشخیص داده شد، حالات: {[states_history[i][-1] if states_history[i] else 'None' for i in range(4)]}")
        detection_log.append({"frame": frame_count, "num_persons": num_persons, "states": [states_history[i][-1] if states_history[i] else None for i in range(4)]})
        
        if frame_count % 10 == 0:
            cv2.imwrite(os.path.join(frames_dir, f"frame_{frame_count}.jpg"), frame)
            debug_frames.append(frame)
        
        cv2.imshow('Debug Frame', frame)
        cv2.waitKey(1)
    
    cap.release()
    cv2.destroyAllWindows()
    
    pd.DataFrame(detection_log).to_csv(os.path.join(output_dir, "detection_log_v2.csv"), index=False)
    
    reports = []
    for idx, states in enumerate(states_history):
        if not states:
            continue
        state_counts = pd.Series(states).value_counts(normalize=True) * 100
        productive_pct = state_counts.get("Productive", 0) + state_counts.get("Standing Moving", 0)
        unproductive_pct = 100 - productive_pct
        report = {
            "فرد": idx + 1,
            "مدت زمان کل (ثانیه)": duration,
            "رزولوشن": f"{width}x{height}",
            "فریم در ثانیه": fps,
            "تعداد فریم‌ها": total_frames,
            "درصد مولد": productive_pct,
            "درصد غیرمولد": unproductive_pct,
            "تفکیک حالات": state_counts.to_dict()
        }
        reports.append(report)
    
    pd.DataFrame(reports).to_csv(os.path.join(output_dir, "analysis_report_v2.csv"), index=False)
    return reports, debug_frames

# تابع داشبورد
def show_dashboard(reports=None, debug_frames=None):
    st.title("📊 داشبورد تحلیلگر عملکرد (چندنفره، چند حالته)")
    
    if reports is None:
        st.header("آپلود ویدئو")
        video_file = st.file_uploader("ویدئو را انتخاب کنید (MP4)", type=["mp4"])
        rotation_option = st.selectbox("جهت چرخش ویدئو", 
                                      ["پادساعتگرد 90 درجه", "ساعتگرد 90 درجه", "180 درجه", "بدون چرخش"],
                                      index=0)
        rotation_map = {
            "پادساعتگرد 90 درجه": cv2.ROTATE_90_COUNTERCLOCKWISE,
            "ساعتگرد 90 درجه": cv2.ROTATE_90_CLOCKWISE,
            "180 درجه": cv2.ROTATE_180,
            "بدون چرخش": None
        }
        rotation_code = rotation_map[rotation_option]
        
        if video_file:
            video_path = os.path.join("videos", "uploaded_video.mp4")
            os.makedirs("videos", exist_ok=True)
            with open(video_path, "wb") as f:
                f.write(video_file.getbuffer())
            st.success("ویدئو آپلود شد. تحلیل در حال انجام...")
            reports, debug_frames = analyze_video(video_path, rotation_code)
    
    if reports:
        st.header(f"تعداد افراد تشخیص‌داده‌شده: {len(reports)}")
        for report in reports:
            st.subheader(f"تحلیل برای فرد {report['فرد']}")
            st.write(f"مدت زمان: {report['مدت زمان کل (ثانیه)']} ثانیه")
            st.write(f"رزولوشن: {report['رزولوشن']}")
            st.write(f"فریم در ثانیه: {report['فریم در ثانیه']}")
            st.write(f"تعداد فریم‌ها: {report['تعداد فریم‌ها']}")
            
            fig = px.pie(values=[report["درصد مولد"], report["درصد غیرمولد"]],
                         names=["مولد", "غیرمولد"],
                         title=f"تفکیک بهره‌وری فرد {report['فرد']}",
                         color_discrete_sequence=["#00CC96", "#EF553B"])
            st.plotly_chart(fig)
            
            state_counts = pd.Series(report["تفکیک حالات"])
            fig_states = px.bar(x=state_counts.index, y=state_counts.values,
                                labels={"x": "حالت", "y": "درصد (%)"},
                                title=f"توزیع حالات فرد {report['فرد']}",
                                color=state_counts.index,
                                color_discrete_map={
                                    "Productive": "#00CC96",
                                    "Sitting Idle": "#EF553B",
                                    "Standing Idle": "#636EFA",
                                    "Standing Moving": "#00CC96",
                                    "Away": "#AB63FA"
                                })
            st.plotly_chart(fig_states)
            
            st.header("گزارش کامل")
            st.table(pd.DataFrame([report]))
    
    if debug_frames:
        st.header("فریم نمونه (دیباگ)")
        frame = debug_frames[0]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st.image(frame_rgb, caption="فریم نمونه با نقاط کلیدی", use_column_width=True)
    
    st.header("ذخیره گزارش")
    if reports and st.button("ذخیره گزارش به JSON"):
        pd.DataFrame(reports).to_json(os.path.join("output", "analysis_report_v2.json"), orient="records", lines=True, force_ascii=False)
        st.success("گزارش به JSON ذخیره شد")
    
    st.header("لاگ تشخیص")
    if os.path.exists(os.path.join("output", "detection_log_v2.csv")):
        detection_log = pd.read_csv(os.path.join("output", "detection_log_v2.csv"))
        st.write("لاگ تعداد افراد و حالات در هر فریم:")
        st.dataframe(detection_log)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="تحلیلگر عملکرد (چندنفره، چند حالته)")
    parser.add_argument("--video", type=str, help="مسیر فایل ویدئو")
    parser.add_argument("--dashboard", action="store_true", help="نمایش داشبورد پس از تحلیل")
    args = parser.parse_args()
    
    if args.video:
        reports, debug_frames = analyze_video(args.video)
        print("تحلیل کامل شد. گزارش در output/analysis_report_v2.csv ذخیره شد")
        print(reports)
        if args.dashboard:
            show_dashboard(reports, debug_frames)
    else:
        show_dashboard()