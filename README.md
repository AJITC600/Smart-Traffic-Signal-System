\# 🚦 Smart Traffic Signal System using YOLOv8



A computer vision-based smart traffic signal system that uses \*\*YOLOv8\*\* to detect and count cars from traffic video. The detected number of cars is used to determine the traffic signal state.



\## 📌 Project Overview



Traffic congestion is a common problem in busy areas. Traditional traffic signals generally operate using fixed timing, which may not respond effectively to changing traffic conditions.



This project demonstrates a simple intelligent traffic signal system using \*\*YOLOv8 object detection\*\*. The system processes an uploaded traffic video, detects cars in each frame, counts the detected cars, and displays a traffic signal based on the number of cars.



\## 🎯 Objective



The main objectives of this project are:



\- Detect cars from traffic video using YOLOv8.

\- Count the number of detected cars.

\- Estimate traffic density based on the car count.

\- Automatically determine a traffic signal state.

\- Display the processed video through a Streamlit web application.



\## 🧠 YOLOv8 Car Detection



The project uses the pretrained \*\*YOLOv8n\*\* model.



The COCO class ID for a car is:



```text

2

