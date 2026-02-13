"""
Enhanced Face Locking module with MQTT servo control
"""
import sys
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, List, Tuple, Any
import cv2
import numpy as np
import mediapipe as mp
import paho.mqtt.client as mqtt

from src.embed import ArcFaceEmbedderONNX, EmbeddingResult

# -------------------------
# Configuration
# -------------------------

DB_PATH = Path(__file__).parent.parent / "data/db/face_db.npz"
HISTORY_DIR = Path(__file__).parent.parent / "data/history"
DISTANCE_THRESHOLD = 0.35
LOCK_RELEASE_FRAMES = 30

# MQTT Configuration
MQTT_BROKER = "157.173.101.159"  # Change to your MQTT broker IP
MQTT_PORT = 1883
MQTT_TOPIC_SERVO_ANGLE = "face_tracking/servo_angle"

# MediaPipe settings
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

# Eye landmarks indices
LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144] 
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
MOUTH_INDICES = [61, 291, 0, 17]

# Alignment indices
LEFT_EYE_CENTER_IDX = 33
RIGHT_EYE_CENTER_IDX = 263
NOSE_IDX = 1
MOUTH_LEFT_IDX = 61
MOUTH_RIGHT_IDX = 291

# Servo Configuration
SERVO_ANGLE_MIN = 0
SERVO_ANGLE_MAX = 180
SERVO_SMOOTHING_FACTOR = 0.3

# -------------------------
# MQTT Controller
# -------------------------

class MQTTServoController:
    """Simple MQTT publisher for servo control."""
    
    def __init__(self, broker=MQTT_BROKER, port=MQTT_PORT):
        self.broker = broker
        self.port = port
        self.topic = MQTT_TOPIC_SERVO_ANGLE
        self.client = mqtt.Client()
        self.connected = False
        
    def connect(self):
        """Connect to MQTT broker."""
        try:
            self.client.connect(self.broker, self.port, 60)
            self.client.loop_start()
            self.connected = True
            print(f"[MQTT] Connected to broker at {self.broker}:{self.port}")
            return True
        except Exception as e:
            print(f"[MQTT] Connection failed: {e}")
            return False
    
    def send_angle(self, angle):
        """Send servo angle via MQTT."""
        if not self.connected:
            print("[MQTT] Not connected to broker")
            return False
        
        try:
            # Clamp angle to 0-180 range
            angle = max(0, min(180, int(angle)))
            
            # Publish angle
            result = self.client.publish(self.topic, str(angle), qos=1)
            
            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                return True
            else:
                return False
        except Exception as e:
            print(f"[MQTT] Error sending angle: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from MQTT broker."""
        if self.connected:
            self.client.loop_stop()
            self.client.disconnect()
            self.connected = False
            print("[MQTT] Disconnected from broker")

# -------------------------
# Database helpers
# -------------------------

def load_database() -> Dict[str, np.ndarray]:
    """Load enrolled face database."""
    if not DB_PATH.exists():
        return {}
    data = np.load(str(DB_PATH), allow_pickle=True)
    db = {}
    for k in data.files:
        emb = np.asarray(data[k], dtype=np.float32)
        if emb.ndim > 1:
            emb = emb.reshape(-1)
        db[k] = emb
    return db

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return float(np.dot(a, b))

def recognize_face(embedding: np.ndarray, db: Dict[str, np.ndarray], threshold: float) -> Tuple[str, float]:
    """Recognize a face by comparing embedding with database."""
    best_name = "Unknown"
    best_similarity = 0.0
    
    for name, ref_emb in db.items():
        similarity = cosine_similarity(embedding, ref_emb)
        if similarity > best_similarity and similarity >= (1.0 - threshold):
            best_similarity = similarity
            best_name = name
    
    return best_name, best_similarity

# -------------------------
# Position Tracking
# -------------------------

class PositionTracker:
    """Tracks face position and converts to servo angles."""
    
    def __init__(self, screen_width: int):
        self.screen_width = screen_width
        self.current_angle = 90
        self.position_buffer = []
        self.buffer_size = 5
    
    def calculate_angle_from_position(self, face_center_x: float) -> float:
        """Calculate servo angle from face position on screen."""
        normalized_x = face_center_x / self.screen_width
        normalized_x = max(0.0, min(1.0, normalized_x))
        
        # Map to servo angle range
        angle = SERVO_ANGLE_MAX - (normalized_x * SERVO_ANGLE_MAX)
        
        # Smoothing
        self.position_buffer.append(angle)
        if len(self.position_buffer) > self.buffer_size:
            self.position_buffer.pop(0)
        
        smoothed_angle = np.mean(self.position_buffer)
        return smoothed_angle
    
    def update(self, face_center_x: Optional[float]) -> Optional[float]:
        """Update position tracker with new face position."""
        if face_center_x is None:
            return None
        
        target_angle = self.calculate_angle_from_position(face_center_x)
        angle_difference = target_angle - self.current_angle
        self.current_angle += angle_difference * SERVO_SMOOTHING_FACTOR
        
        return self.current_angle

# -------------------------
# MediaPipe Face Detector
# -------------------------

class MediaPipeFaceDetector:
    """Face detector using MediaPipe Face Mesh."""
    
    def __init__(self, min_size=(50, 50)):
        self.min_size = min_size
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=5,
            refine_landmarks=True,
            min_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE
        )
    
    def detect(self, frame: np.ndarray, max_faces: int = 5) -> List[Dict[str, Any]]:
        """Detect faces using MediaPipe and return face data."""
        faces = []
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        results = self.face_mesh.process(rgb_frame)
        rgb_frame.flags.writeable = True
        
        if results.multi_face_landmarks:
            h, w = frame.shape[:2]
            
            for face_landmarks in results.multi_face_landmarks:
                landmarks = []
                for lm in face_landmarks.landmark:
                    landmarks.append([lm.x * w, lm.y * h, lm.z * w])
                landmarks = np.array(landmarks)
                
                x_min = int(np.min(landmarks[:, 0]))
                y_min = int(np.min(landmarks[:, 1]))
                x_max = int(np.max(landmarks[:, 0]))
                y_max = int(np.max(landmarks[:, 1]))
                
                width = x_max - x_min
                height = y_max - y_min
                if width < self.min_size[0] or height < self.min_size[1]:
                    continue
                
                face_data = {
                    'x1': x_min, 'y1': y_min, 'x2': x_max, 'y2': y_max,
                    'landmarks': landmarks,
                    'width': width, 'height': height
                }
                faces.append(face_data)
                
                if len(faces) >= max_faces:
                    break
        
        return faces

def align_face_mediapipe(frame: np.ndarray, landmarks: np.ndarray, out_size: Tuple[int, int] = (112, 112)):
    """Align face using MediaPipe landmarks for 5-point alignment."""
    if landmarks is None or len(landmarks) < 5:
        return None
    
    src_points = np.array([
        landmarks[LEFT_EYE_CENTER_IDX][:2],
        landmarks[RIGHT_EYE_CENTER_IDX][:2],
        landmarks[NOSE_IDX][:2],
        landmarks[MOUTH_LEFT_IDX][:2],
        landmarks[MOUTH_RIGHT_IDX][:2]
    ], dtype=np.float32)
    
    dst_points = np.array([
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041]
    ], dtype=np.float32)
    
    transform = cv2.estimateAffinePartial2D(src_points, dst_points, method=cv2.RANSAC)[0]
    if transform is None:
        return None
    
    aligned_face = cv2.warpAffine(frame, transform, out_size, borderValue=0.0)
    return aligned_face

# -------------------------
# Face Locker
# -------------------------

class FaceLocker:
    """Locks onto a specific face and tracks it with servo control."""
    
    def __init__(self, target_name: str, target_embedding: np.ndarray, db: Dict[str, np.ndarray]):
        self.target_name = target_name
        self.target_embedding = target_embedding
        self.db = db
        
        self.locked = False
        self.fail_count = 0
        self.total_lock_frames = 0
        
        # Initialize MQTT servo controller
        self.servo_controller = MQTTServoController()
        self.servo_controller.connect()
        
        # MediaPipe detector and embedder
        self.detector = MediaPipeFaceDetector(min_size=(50, 50))
        self.embedder = ArcFaceEmbedderONNX(
            model_path=str(Path(__file__).parent.parent / "models/embedder_arcface.onnx"),
            debug=False,
        )
        
        # Position tracker
        self.position_tracker = None
        
        print(f"[MQTT] Ready to send servo commands for {target_name}")
    
    def set_screen_size(self, width: int):
        """Initialize position tracker with screen width."""
        self.position_tracker = PositionTracker(width)
    
    def recognize_all_faces(self, frame: np.ndarray, faces) -> List[Tuple[dict, str, float]]:
        """Recognize all faces in the frame."""
        results = []
        
        for face_data in faces:
            aligned = align_face_mediapipe(frame, face_data['landmarks'])
            if aligned is None:
                continue
            
            res: EmbeddingResult = self.embedder.embed(aligned)
            identity, similarity = recognize_face(res.embedding, self.db, DISTANCE_THRESHOLD)
            
            center_x = (face_data['x1'] + face_data['x2']) / 2.0
            center_y = (face_data['y1'] + face_data['y2']) / 2.0
            
            face_data_full = {
                'face_data': face_data,
                'center_x': center_x,
                'center_y': center_y,
                'width': face_data['width'],
                'height': face_data['height']
            }
            
            results.append((face_data_full, identity, similarity))
        
        return results
    
    def find_target_face(self, recognized_faces: List[Tuple[dict, str, float]]) -> Tuple[Optional[dict], float]:
        """Find the target face among recognized faces."""
        for face_data, identity, similarity in recognized_faces:
            if identity == self.target_name:
                distance = 1.0 - similarity
                return face_data, distance
        
        return None, 1.0
    
    def update_position_tracking(self, target_face_data: Optional[Dict]) -> Optional[float]:
        """Update position tracking and return servo angle."""
        if self.position_tracker and target_face_data:
            face_center_x = target_face_data.get('center_x')
            servo_angle = self.position_tracker.update(face_center_x)
            
            # Send angle via MQTT
            if servo_angle is not None:
                self.servo_controller.send_angle(servo_angle)
            
            return servo_angle
        return None
    
    def close(self):
        """Close the face locker and MQTT controller."""
        self.servo_controller.disconnect()

# -------------------------
# Main function
# -------------------------

def main():
    """Enhanced Face Locking main function with MQTT servo control."""
    
    # Load database
    db = load_database()
    if not db:
        print("ERROR: No enrolled identities. Run: python -m src.enroll")
        return False
    
    # Show available identities
    names = sorted(db.keys())
    print("\nEnrolled identities:")
    for i, name in enumerate(names, 1):
        print(f"  {i}. {name}")
    
    # Let user choose target
    print("\nEnter the name of the identity to lock (exact match): ", end="")
    try:
        choice = input().strip()
    except EOFError:
        choice = names[0] if names else ""
    
    if not choice:
        choice = names[0] if names else ""
    
    if choice not in db:
        print(f"ERROR: '{choice}' not in database. Choose from: {names}")
        return False
    
    print(f"Will lock onto: {choice}")
    
    # Initialize face locker
    locker = FaceLocker(choice, db[choice], db)
    
    # Open camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open camera.")
        return False
    
    # Get screen dimensions
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    locker.set_screen_size(width)
    
    print(f"\n" + "="*50)
    print(f"Face Locking with Servo Control")
    print(f"Target: {choice.upper()}")
    print(f"Screen: {width}x{height}")
    print(f"Servo Range: {SERVO_ANGLE_MIN}° to {SERVO_ANGLE_MAX}°")
    print(f"MQTT Topic: {MQTT_TOPIC_SERVO_ANGLE}")
    print(f"Press 'q' to quit")
    print("="*50 + "\n")
    
    # Performance tracking
    t0 = time.time()
    frames = 0
    fps = 0.0
    
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            
            frames += 1
            dt = time.time() - t0
            if dt >= 1.0:
                fps = frames / dt
                frames = 0
                t0 = time.time()
            
            H, W = frame.shape[:2]
            vis = frame.copy()
            
            # Detect faces
            faces = locker.detector.detect(frame, max_faces=5)
            
            # Recognize faces
            recognized_faces = locker.recognize_all_faces(frame, faces)
            
            # Find target face
            target_face, target_distance = locker.find_target_face(recognized_faces)
            
            # Update position tracking and send servo commands
            servo_angle = None
            if target_face:
                servo_angle = locker.update_position_tracking(target_face)
            
            if not locker.locked:
                # Looking for target
                if target_face is not None and target_distance <= DISTANCE_THRESHOLD:
                    locker.locked = True
                    locker.fail_count = 0
                    locker.total_lock_frames = 0
                    print(f"[LOCKED] Target locked at ({int(target_face['center_x'])}, {int(target_face['center_y'])})")
            
            else:
                # Already locked, track target
                if target_face is not None and target_distance <= DISTANCE_THRESHOLD:
                    # Still locked on target
                    locker.fail_count = 0
                    locker.total_lock_frames += 1
                    face = target_face['face_data']
                    
                    # Draw locked face
                    cv2.rectangle(vis, (face['x1'], face['y1']), (face['x2'], face['y2']), (0, 255, 0), 2)
                    
                    # Draw crosshair
                    center_x = (face['x1'] + face['x2']) // 2
                    center_y = (face['y1'] + face['y2']) // 2
                    cv2.line(vis, (center_x, face['y1']), (center_x, face['y2']), (0, 255, 0), 1)
                    cv2.line(vis, (face['x1'], center_y), (face['x2'], center_y), (0, 255, 0), 1)
                    
                    # Show info
                    cv2.putText(vis, f"LOCKED: {locker.target_name.upper()}", 
                               (face['x1'], max(0, face['y1'] - 25)),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    if servo_angle:
                        angle_text = f"Servo: {servo_angle:.1f}°"
                        cv2.putText(vis, angle_text, 
                                   (face['x1'], face['y2'] + 20),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    
                    # Terminal output
                    if frames % 10 == 0:
                        pos_x, pos_y = int(target_face['center_x']), int(target_face['center_y'])
                        angle_str = f"{servo_angle:.1f}°" if servo_angle else "N/A"
                        print(f"[SCANNING] {locker.target_name.upper()} | Pos: ({pos_x},{pos_y}) | Angle: {angle_str}")
                else:
                    # Lost target
                    locker.fail_count += 1
                    if locker.fail_count >= LOCK_RELEASE_FRAMES:
                        locker.locked = False
                        locker.fail_count = 0
                        print(f"[LOST] Lost signal on {locker.target_name.upper()}")
            
            # Draw other faces
            for face_data, identity, similarity in recognized_faces:
                face = face_data['face_data']
                
                if locker.locked and identity == locker.target_name:
                    continue
                
                if identity == "Unknown":
                    color = (0, 0, 255)
                elif identity == locker.target_name and not locker.locked:
                    color = (0, 255, 0)
                else:
                    color = (255, 255, 0)
                
                cv2.rectangle(vis, (face['x1'], face['y1']), (face['x2'], face['y2']), color, 2)
                
                label = identity if identity != "Unknown" else f"Unknown ({similarity:.2f})"
                cv2.putText(vis, label, 
                           (face['x1'], max(0, face['y1'] - 5)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Draw status
            locked_status = "LOCKED" if locker.locked else "Searching..."
            if servo_angle:
                status = f"Target: {locker.target_name} | {locked_status} | Angle: {servo_angle:.1f}° | FPS: {fps:.1f}"
            else:
                status = f"Target: {locker.target_name} | {locked_status} | FPS: {fps:.1f}"
            cv2.putText(vis, status, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.putText(vis, "Press 'q' to quit", (10, H - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            
            # Show frame
            cv2.imshow("Face Locking with Servo Control", vis)
            
            # Handle keypress
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
    
    finally:
        # Clean up
        cap.release()
        cv2.destroyAllWindows()
        locker.close()
    
    print("\nFace Locking ended.")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)