# LiveFaceMonitoring

Lightweight folder for running the live face-monitoring frontend, Python controller, and servo Arduino sketch.

Contents
- `faceLockServo.py` — Python controller to run live face monitoring and (optionally) control a servo.
- `frontend.html` — Simple web frontend to display camera / control UI.
- `servo_motor/servo_motor.ino` — Arduino sketch for a single servo.

Prerequisites
- Python 3.8+ installed.
- From the project root, install Python dependencies (if present):

  ```bash
  pip install -r requirements.txt
  ```

- Arduino IDE (or compatible) to upload `servo_motor/servo_motor.ino` to a board.

Quick run (recommended)
1. Open a terminal at the project root (one level above this folder).
2. Start the Python controller (uses package module path):

  ```bash
  python -m LiveFaceMonitoring.faceLockServo
  ```

  - This runs the live face-monitoring/servo controller. If the script requires specific serial/port settings or arguments, edit the script or pass them as the script accepts.

Frontend (viewing the UI)
- Option A — Open directly: double-click `frontend.html` to open in your browser. Note: some browsers restrict camera access for local file URLs.
- Option B — Serve locally (recommended to avoid CORS / file access issues):

  ```bash
  cd LiveFaceMonitoring
  python -m http.server 8000
  ```

  Then open: http://localhost:8000/frontend.html

Arduino / Servo
- Open `servo_motor/servo_motor.ino` in the Arduino IDE and upload to your board.
- Wiring (general):
  - Servo VCC -> 5V (or external 5V supply for high torque)
  - Servo GND -> GND (common ground with controller PC/Arduino)
  - Servo signal -> PWM pin (e.g., D9) — adjust in the sketch if needed
- Power note: use a separate power source for large servos; ensure grounds are common.

Integration notes
- Connect the Arduino to the computer via USB if `faceLockServo.py` expects serial communication. Ensure the script's serial port matches the Arduino COM/tty device.
- If the Python controller accesses the camera, allow camera permission in your browser (for the frontend) or ensure the camera is available to the Python process.

Troubleshooting
- If the camera doesn't appear in the browser, try serving the HTML (see Option B) or use a different browser.
- If servo doesn't respond, check COM port, baud rate (match Arduino sketch), and common ground.

Want me to run or test any part of this now? I can try launching the Python controller or serve the frontend for you.
