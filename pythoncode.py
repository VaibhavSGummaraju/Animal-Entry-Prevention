PYTHON CODE


import cv2
import numpy as np
import matplotlib.pyplot as plt
import serial
import time

# Initialize serial communication
ser = serial.Serial('COM5', 9600)  # Replace 'COM5' with your Arduino's serial port

# Load the pre-trained model
net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "mobilenet_iter_73000.caffemodel")

# Initialize webcam video capture
cap = cv2.VideoCapture(0)
plt.ion()  # Turn on interactive mode
fig, ax = plt.subplots()

def press(event):
    if event.key == 'q':
        plt.close(fig)
        cap.release()
        ser.close()

fig.canvas.mpl_connect('key_press_event', press)

while cap.isOpened():
    # Check if "motion" signal is received from Arduino
    if ser.in_waiting > 0:
        motion_signal = ser.readline().decode().strip()
        if motion_signal == "motion":
            ret, frame = cap.read()
            if not ret:
                break

            (h, w) = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
            net.setInput(blob)
            detections = net.forward()

            max_confidence = 0
            human_detected = False  # Flag to track if any human is detected

            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.5:
                    idx = int(detections[0, 0, i, 1])
                    if idx == 15:  # Class ID for 'person'
                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        (startX, startY, endX, endY) = box.astype("int")
                        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                        label = f"Person: {confidence:.2f}"
                        cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        if confidence > max_confidence:
                            max_confidence = confidence

                        # Mark that a human was detected
                        human_detected = True

            if human_detected:
                print("human detected")
            else:
                print("human not detected")

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            ax.clear()
            ax.imshow(frame_rgb)
            plt.pause(0.001)

            # Send the highest confidence level to Arduino
            ser.write(f"{max_confidence:.2f}\n".encode())
            time.sleep(0.1)  # Delay to avoid overwhelming the serial buffer

cap.release()
plt.close()
