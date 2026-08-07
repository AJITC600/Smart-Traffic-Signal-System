import streamlit as st
import cv2
import tempfile
from ultralytics import YOLO

st.set_page_config(page_title="Smart Traffic Signal", layout="wide")

st.title("🚦 Smart Traffic Signal using YOLOv8")

uploaded_file = st.file_uploader(
    "Upload a traffic video",
    type=["mp4", "avi", "mov"]
)

if uploaded_file is not None:

    # Save uploaded video
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    # Load YOLO model
    model = YOLO("yolov8n.pt")

    # Open uploaded video
    cap = cv2.VideoCapture(tfile.name)

    stframe = st.empty()

    # COCO class IDs
    vehicle_classes = [2]   # car only
    # vehicle_classes = [2,3,5,7]  # Uncomment for car,bike,bus,truck

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            break

        results = model(frame)

        vehicles = []
        count = 0

        for result in results:

            for box in result.boxes:

                cls = int(box.cls[0])

                if cls in vehicle_classes:

                    count += 1

                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    vehicles.append((x1, y1, x2, y2))

        # Traffic Signal Logic
        if count <= 5:
            signal = "GREEN"
            color = (0,255,0)

        elif count <= 10:
            signal = "YELLOW"
            color = (0,255,255)

        else:
            signal = "RED"
            color = (0,0,255)

        # Draw boxes
        for (x1,y1,x2,y2) in vehicles:
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

        # Vehicle count
        cv2.putText(
            frame,
            f"Cars: {count}",
            (20,40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2
        )

        # Signal
        cv2.putText(
            frame,
            f"Signal: {signal}",
            (20,80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2
        )

        # Traffic light box
        cv2.rectangle(frame,(500,20),(560,170),(50,50,50),-1)

        red = (0,0,100)
        yellow = (0,100,100)
        green = (0,100,0)

        if signal == "RED":
            red = (0,0,255)

        elif signal == "YELLOW":
            yellow = (0,255,255)

        else:
            green = (0,255,0)

        cv2.circle(frame,(530,50),15,red,-1)
        cv2.circle(frame,(530,95),15,yellow,-1)
        cv2.circle(frame,(530,140),15,green,-1)

        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Show frame
        stframe.image(frame, channels="RGB", use_container_width=True)

    cap.release()

    st.success("Video processing completed!")