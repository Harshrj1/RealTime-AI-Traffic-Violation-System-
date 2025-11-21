import cv2
import numpy as np
import time
import csv
from sort import Sort  


VIDEO_PATH = r"C:\Users\samiy\OneDrive\Desktop\RealTime_Traffic_Violation_Detection\data\sample_traffic.mp4"
violation_log = r"C:\Users\samiy\OneDrive\Desktop\RealTime_Traffic_Violation_Detection\violation_log.csv"

cap = cv2.VideoCapture(VIDEO_PATH)
tracker = Sort()


light_state = "GREEN"
last_switch_time = time.time()
switch_interval = 10  


speed_limit = 60  
positions = {}
frame_rate = cap.get(cv2.CAP_PROP_FPS)
distance_per_pixel = 0.05 


with open(violation_log, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Time", "Violation_Type", "Vehicle_ID", "Details"])


while True:
    ret, frame = cap.read()
    if not ret:
        break

    
    frame = cv2.resize(frame, (960, 540))
    height, width, _ = frame.shape

    
    current_time = time.time()
    if current_time - last_switch_time > switch_interval:
        light_state = "RED" if light_state == "GREEN" else "GREEN"
        last_switch_time = current_time

    
    color = (0, 255, 0) if light_state == "GREEN" else (0, 0, 255)
    cv2.circle(frame, (80, 80), 30, color, -1)
    cv2.putText(frame, f"LIGHT: {light_state}", (40, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    detections = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 60 and h > 60:
            detections.append([x, y, x + w, y + h, 0.9])

    detections = np.array(detections)
    tracked_objects = tracker.update(detections)

    
    stop_line_y = int(height * 0.65)
    cv2.line(frame, (0, stop_line_y), (width, stop_line_y), (255, 255, 0), 2)

    for obj in tracked_objects:
        x1, y1, x2, y2, obj_id = map(int, obj)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        # Draw object box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
        cv2.putText(frame, f"ID {obj_id}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

        if light_state == "RED" and cy < stop_line_y:
            cv2.putText(frame, "RED LIGHT VIOLATION!", (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            with open(violation_log, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([time.strftime("%H:%M:%S"), "Red Light", obj_id, f"Crossed at {cy}"])

    
    for obj in tracked_objects:
        x1, y1, x2, y2, obj_id = map(int, obj)
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        if obj_id in positions:
            prev_x, prev_y, prev_time = positions[obj_id]
            distance = np.sqrt((cx - prev_x) ** 2 + (cy - prev_y) ** 2)
            time_diff = current_time - prev_time
            speed = (distance * distance_per_pixel) / (time_diff + 1e-5) * 3.6  # km/h

            if speed > speed_limit:
                cv2.putText(frame, f"Speed: {int(speed)} km/h (OVERSPEED!)", (x1, y2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                with open(violation_log, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([time.strftime("%H:%M:%S"), "Overspeed", obj_id, f"{int(speed)} km/h"])
            else:
                cv2.putText(frame, f"Speed: {int(speed)} km/h", (x1, y2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        positions[obj_id] = (cx, cy, current_time)

    # ==============================
    # HELMET & SEATBELT DETECTION (APPROXIMATION)
    # ==============================
    # Detect human-like contours near the top
    roi = frame[int(height * 0.3):int(height * 0.8), :]
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, roi_thresh = cv2.threshold(gray_roi, 180, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(roi_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if 50 < w < 120 and 100 < h < 250:
            cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 255), 2)
            # Approx: Random helmet status toggle
            helmet_on = np.random.choice([True, False], p=[0.7, 0.3])
            seatbelt_on = np.random.choice([True, False], p=[0.8, 0.2])

            status = []
            if not helmet_on:
                status.append("No Helmet")
            if not seatbelt_on:
                status.append("No Seatbelt")

            if status:
                cv2.putText(roi, ", ".join(status), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                with open(violation_log, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([time.strftime("%H:%M:%S"), " / ".join(status), "-", "Approx Detected"])

    # ==============================
    # DISPLAY FRAME
    # ==============================
    cv2.imshow("Real-Time Traffic Violation System", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
