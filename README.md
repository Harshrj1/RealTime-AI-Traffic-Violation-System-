🚦 Real-Time AI Traffic Violation Detection System
This project focuses on detecting common traffic violations in real time using computer vision and deep learning. The goal is to automate traffic monitoring so that violations can be identified without relying on manual checking or physical patrol.

The system processes live video (or recorded footage) and automatically detects:

1. Helmetless riding
2. Overspeeding
3. Red-light jumping
4. Vehicle number plates (for logging violations)

The project combines object detection, tracking, and OCR to create a simple, end-to-end pipeline for road safety monitoring.
 ✨ Main Features
✔ Helmet Detection : 
Checks whether a rider is wearing a helmet and flags violations.


✔ Speed Estimation : 
Tracks vehicles across frames and estimates their speed. If a vehicle crosses the defined speed limit, it is marked as overspeeding.


✔ Red-Light Monitoring : 
Detects the state of the traffic signal and checks if a vehicle crosses the stop line during a red light.


✔ Lane & Wrong-Way Detection : 
Defines regions of interest and identifies if a vehicle changes lanes illegally or enters a lane in the wrong direction.


✔ Number Plate Recognition : 
Detects license plates and extracts the plate number using OCR.

🛠 Tech Used

1.Python


2.OpenCV


3.YOLOv8 / YOLOv5


4.SORT / DeepSORT for tracking


5.Tesseract / EasyOCR


6.NumPy, Pandas





