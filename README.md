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

- Note: For detailed steps and the implementation used to build `faceLockServo.py`, see the FaceLocking project: https://github.com/forgiveness-77/FaceLocking

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

Repository notes
----------------

- System description: This repository contains the PC-side face recognition and tracking controller (`faceLockServo.py`), a simple web live dashboard (`frontend.html`), and an Arduino servo sketch (`servo_motor/servo_motor.ino`). The Python controller uses MediaPipe for face detection, an ArcFace ONNX embedder for recognition, and publishes servo angle commands via MQTT.

- MQTT topics used:
  - `face_tracking/servo_angle` — published by `faceLockServo.py` to instruct servo position (payload: integer angle 0-180).

- Default MQTT broker (as set in the controller): `157.173.101.159:1883` — edit `faceLockServo.py` to change broker/port.

- Live dashboard URL:
  - If you serve the folder locally using `python -m http.server 8000`, open:

    http://localhost:8000/frontend.html

What this repo includes
-----------------------

- PC face recognition/tracking code: `faceLockServo.py`
- Web Live Dashboard: `frontend.html`
- Servo Arduino sketch: `servo_motor/servo_motor.ino`

Missing / Optional components
-----------------------------

- ESP8266 firmware: not included. If you need an ESP8266-based client (instead of the Arduino sketch), tell me target pinout and MQTT expectations and I can add an example firmware sketch that subscribes/publishes to the MQTT topics above.

Publishing this repository publicly
----------------------------------

I don't currently have a public GitHub repository for this workspace. I can prepare the local repository content (done here) and guide you to publish it. Run these commands from the project root to create a public repo and push:

```bash
git init
git add .
git commit -m "Initial commit: LiveFaceMonitoring"
# Create a public repo on GitHub (replace <your-repo-url> after creating the repo manually)
git remote add origin <your-repo-url>
git branch -M main
git push -u origin main
```

If you want, I can help create a GitHub repository for you and push these files — I would need permission (a GitHub token or Oauth flow) to do that on your behalf. Otherwise follow the manual steps above.

If you'd like I can also:
- Pin dependency versions in `requirements.txt`.
- Add an ESP8266 example firmware that listens/publishes MQTT messages.
- Create a sample GitHub Actions workflow for linting/tests.

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
