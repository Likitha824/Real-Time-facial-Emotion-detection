import tensorflow as tf
from keras.models import load_model
import cv2
import numpy as np
import sys

# Load the pre-trained model
print("loading model..")
name= "best_model_CNN.keras"
model = load_model(name)
print("Loaded model from disk")

# Emotion dictionary
emotion_dict = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprise"}

# Face detection model
face_detector = cv2.CascadeClassifier('haarcascades.xml')

# Define the mode parameter: 'live' for webcam or 'video' for processing a video file
mode = input("Enter mode (live/video): ").strip().lower()

if mode == 'video':
    input_video_path = input("Enter input video file path: ").strip()  # User specifies the input file
    output_video_path = 'output_' +name+"_"+ input_video_path.split('/')[-1]  # Automatically generate output filename

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("Error: Cannot open video file.")
        sys.exit()

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            roi_gray_frame = gray_frame[y:y+h, x:x+w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

            # Predict the emotion
            cropped_img = cropped_img.astype(np.float32) / 255.0
            prediction = model.predict(cropped_img)
            maxindex = int(np.argmax(prediction))

            # Draw rectangle and text
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
            cv2.putText(frame, emotion_dict[maxindex], (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        # Write frame to output video
        out.write(frame)

    # Release resources
    cap.release()
    out.release()
    print(f"Output video saved at: {output_video_path}")

elif mode == 'live':
    cap = cv2.VideoCapture(0)  # Use 0 for default webcam
    if not cap.isOpened():
        print("Error: Cannot open webcam.")
        sys.exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            roi_gray_frame = gray_frame[y:y+h, x:x+w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

            # Predict the emotion
            cropped_img = cropped_img.astype(np.float32) / 255.0
            prediction = model.predict(cropped_img)
            maxindex = int(np.argmax(prediction))

            # Draw rectangle and text
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
            cv2.putText(frame, emotion_dict[maxindex], (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow('Emotion Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

else:
    print("Invalid mode! Please choose 'live' or 'video'.")
