import cv2
import streamlit as st
import numpy as np
import time
import pandas as pd
import requests
from deepface import DeepFace

if 'capture_running' not in st.session_state:
    st.session_state.capture_running = False
if 'emotion_log' not in st.session_state:
    st.session_state.emotion_log = []

API_ENDPOINT = "https://your-server.com/upload"  

def analyze_frame(frame):
    result = DeepFace.analyze(img_path=frame, actions=['age', 'gender', 'race', 'emotion'],
                              enforce_detection=False,
                              detector_backend="opencv",
                              align=True,
                              silent=False)
    return result

def overlay_text_on_frame(frame, texts):
    overlay = frame.copy()
    alpha = 0.9  
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 100), (255, 255, 255), -1)  
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    text_position = 15  
    for text in texts:
        cv2.putText(frame, text, (10, text_position), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        text_position += 20

    return frame

def log_emotion(emotion_buffer):
    if emotion_buffer:
        emotions = [e["dominant_emotion"] for e in emotion_buffer]

        most_common_emotion = max(set(emotions), key=emotions.count)

        start_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(int(emotion_buffer[0]['timestamp'])))
        end_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(int(emotion_buffer[-1]['timestamp'])))

        emotion_log = {
            "start_time": start_time,
            "end_time": end_time,
            "dominant_emotion": most_common_emotion
        }

        st.session_state.emotion_log.append(emotion_log)

        send_to_server(emotion_log)

def send_to_server(emotion_log):
    try:
        response = requests.post(API_ENDPOINT, json=emotion_log)
        if response.status_code == 200:
            st.success("Emotion log sent successfully!")
        else:
            st.warning("Failed to send emotion log to server.")
    except Exception as e:
        st.error(f"Error sending log: {e}")

def facesentiment():
    cap = cv2.VideoCapture(0)
    stframe = st.image([])  
    emotion_buffer = []

    start_time = time.time()

    while st.session_state.capture_running:
        ret, frame = cap.read()
        if not ret:
            st.warning("Failed to access webcam")
            break

        result = analyze_frame(frame)
        current_time = time.time()

        result[0]['timestamp'] = current_time

        emotion_buffer.append(result[0])

        face_coordinates = result[0]["region"]
        x, y, w, h = face_coordinates['x'], face_coordinates['y'], face_coordinates['w'], face_coordinates['h']
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        text = f"{result[0]['dominant_emotion']}"
        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        texts = [
            f"Age: {result[0]['age']}",
            f"Face Confidence: {round(result[0]['face_confidence'], 3)}",
            f"Gender: {result[0]['dominant_gender']} {round(result[0]['gender'][result[0]['dominant_gender']], 3)}",
            f"Race: {result[0]['dominant_race']}",
            f"Dominant Emotion: {result[0]['dominant_emotion']} {round(result[0]['emotion'][result[0]['dominant_emotion']], 1)}",
        ]
        frame_with_overlay = overlay_text_on_frame(frame_rgb, texts)

        stframe.image(frame_with_overlay, channels="RGB")

        if current_time - start_time >= 60:
            log_emotion(emotion_buffer)
            emotion_buffer.clear()
            start_time = current_time  

        time.sleep(0.05)  

    cap.release()

def download_log():
    if st.session_state.emotion_log:
        df = pd.DataFrame(st.session_state.emotion_log)
        csv = df.to_csv(index=False)
        st.download_button("Download Emotion Log", csv, "emotion_log.csv", "text/csv")
    else:
        st.warning("No emotion log available yet. Start capture first!")

def main():
    st.title("Real-Time Face Emotion Detection Application")

    activities = ["Webcam Face Detection", "Download Log", "About"]
    choice = st.sidebar.selectbox("Select Activity", activities)

    st.sidebar.markdown("Developed by ECS Team VIT-AP")

    if choice == "Webcam Face Detection":
        html_temp_home1 = """<div style="background-color:
            <h4 style="color:white;text-align:center;">
            Real-time face emotion recognition of webcam feed using OpenCV, DeepFace, and Streamlit.</h4>
            </div>
            </br>"""
        st.markdown(html_temp_home1, unsafe_allow_html=True)

        if st.button("Start/Stop Capture"):
            st.session_state.capture_running = not st.session_state.capture_running
            if st.session_state.capture_running:
                facesentiment()

    elif choice == "Download Log":
        download_log()

    elif choice == "About":
        st.subheader("About this app")
        html_temp4 = """<div style="background-color:
            <h4 style="color:white;text-align:center;">ECS Project Team.</h4>
            <h4 style="color:white;text-align:center;">Thanks for Visiting</h4>
            </div>
            <br></br>
            <br></br>"""
        st.markdown(html_temp4, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
