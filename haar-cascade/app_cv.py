import streamlit as st
import cv2
import tempfile

st.title("Smart traffic signal System")
st.write("Car detection using haar cascade")

uploaded_file = st.file_uploader(
    "Upload traffic Video",
    type = ["mp4", "avi"]
)

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    car_cascade = cv2.CascadeClassifier("cars.xml")
    if car_cascade.empty():
        st.error("Error loading cascade classifier xml file") 
    cap = cv2.VideoCapture(tfile.name)
    stframe = st.empty()
    while True:
      ret, frame = cap.read()
      if not ret:
         break
      gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      cars = car_cascade.detectMultiScale(gray,
                                       scaleFactor=1.1,
                                       minNeighbors=2,
                                       minSize=(30,30))
      count = len(cars)
      if count <= 5:
        signal = "GREEN"
        active = "green"
        color = (0,255,0)
      elif count <= 10:
        signal = "YELLOW"
        active = "yellow"
        color = (0,255,255)
      else:
        signal  = "RED"
        active = "red"
        color = (0,0,255)
      cv2.putText(frame, "Cars: " + str(count), (20,40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
      cv2.putText(frame, "Signal: " + signal, (20,80), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, color,2)
      for (x,y,w,h) in cars:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
      frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
      )
      stframe.image(
        frame,
        channels="RGB",
        use_container_width=True
      )
cap.release()