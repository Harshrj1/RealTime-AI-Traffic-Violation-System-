📝 Project Overview
The Real-Time AI Traffic Violation Detection System is a computer-vision–based project designed to automatically detect common traffic violations using live video feed or CCTV footage. This system leverages deep learning, object detection, and OCR to identify vehicles, track movement, and detect violations such as:
❌ Helmetless riding
❌ Overspeeding
❌ Red-light jumping

The goal is to automate traffic monitoring, reduce manual intervention, and assist law enforcement agencies in enforcing road safety.
🧠 Key Features
🔍 Real-Time Object Detection

Uses YOLOv8 / YOLOv5 (or your chosen model) to detect:
1.Vehicles
2.People
3.Helmets
4.Traffic lights

🎥 Live Video Processing
Works on CCTV cameras or real-time webcam feeds

👤 Helmet Detection
Classifies whether a rider is wearing a helmet

🚗 Overspeed Detection
Tracks vehicle movement using centroid tracking
Calculates speed using distance per frame conversion
Alerts when the vehicle exceeds speed limits

🚦 Red-Light Violation Detection
Detects traffic light state (RED/YELLOW/GREEN)
Identifies if a vehicle crossed the stop line during red light


Detects lane departures or wrong-way driving

🔤 License Plate Recognition (ANPR/OCR)

Extracts plate region

Applies OCR (Tesseract / EasyOCR)

Stores plate number linked with violation
