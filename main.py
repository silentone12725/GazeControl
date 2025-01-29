import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import time
from collections import deque
import math
import torch
from torch import cuda

# Disable pyautogui fail-safe for smoother operation
pyautogui.FAILSAFE = False

class TorchNoseTracker:
    def __init__(self):
        # Initialize CUDA device
        self.device = torch.device('cuda' if cuda.is_available() else 'cpu')
        if not cuda.is_available():
            print("Warning: CUDA not available, falling back to CPU")
        
        # Initialize mediapipe face mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Initialize webcam
        self.cap = cv2.VideoCapture(0)
        
        # Get screen dimensions
        self.screen_width, self.screen_height = pyautogui.size()
        
        # Smoothing parameters
        self.smooth_factor = 0.8
        self.prev_x = torch.tensor([self.screen_width // 2], device=self.device)
        self.prev_y = torch.tensor([self.screen_height // 2], device=self.device)
        
        # Calibration values
        self.calibrated = False
        self.cal_left = None
        self.cal_right = None
        self.cal_top = None
        self.cal_bottom = None
        
        # MediaPipe eye landmarks
        self.LEFT_EYE_LANDMARKS = torch.tensor([362, 385, 387, 263, 373, 380], device=self.device)
        self.RIGHT_EYE_LANDMARKS = torch.tensor([33, 160, 158, 133, 153, 144], device=self.device)
        
        # Nose landmark index
        self.NOSE_TIP = 4
        
        # Blink detection parameters
        self.BLINK_RATIO_THRESHOLD = 4.5
        self.blink_history = deque(maxlen=10)
        self.last_blink_time = time.time()
        self.last_click_time = time.time()
        self.BLINK_TIMEOUT = 1.0
        self.CLICK_COOLDOWN = 1.0

    def to_tensor(self, landmarks):
        """Convert MediaPipe landmarks to PyTorch tensor"""
        points = [[lm.x, lm.y, lm.z] for lm in landmarks.landmark]
        return torch.tensor(points, device=self.device)

    @torch.inference_mode()
    def get_blink_ratio(self, eye_points, landmarks_tensor):
        """Calculate eye aspect ratio using PyTorch"""
        eye_landmarks = landmarks_tensor[eye_points]
        
        # Get corner points
        corner_left = eye_landmarks[0, :2]
        corner_right = eye_landmarks[3, :2]
        
        # Get midpoints of top and bottom
        top_points = eye_landmarks[1:3, :2].mean(dim=0)
        bottom_points = eye_landmarks[4:6, :2].mean(dim=0)

        # Calculate distances
        horizontal_length = torch.norm(corner_right - corner_left)
        vertical_length = torch.norm(bottom_points - top_points)
        
        ratio = horizontal_length / (vertical_length + 1e-7)
        return ratio.item()

    @torch.inference_mode()
    def detect_blink(self, landmarks):
        """Detect blink using PyTorch calculations"""
        landmarks_tensor = self.to_tensor(landmarks)
        
        left_eye_ratio = self.get_blink_ratio(self.LEFT_EYE_LANDMARKS, landmarks_tensor)
        right_eye_ratio = self.get_blink_ratio(self.RIGHT_EYE_LANDMARKS, landmarks_tensor)
        blink_ratio = (left_eye_ratio + right_eye_ratio) / 2
        
        return blink_ratio > self.BLINK_RATIO_THRESHOLD

    def process_blinks(self):
        """Process blink history and perform clicks"""
        current_time = time.time()
        
        # Clean up old blinks
        while self.blink_history and current_time - self.blink_history[0] > self.BLINK_TIMEOUT:
            self.blink_history.popleft()
        
        if current_time - self.last_click_time > self.CLICK_COOLDOWN:
            if len(self.blink_history) >= 2:
                pyautogui.click(button='right')
                self.last_click_time = current_time
                self.blink_history.clear()
                return "Right Click"
            elif len(self.blink_history) == 1:
                if current_time - self.blink_history[0] > 0.5:
                    pyautogui.click(button='left')
                    self.last_click_time = current_time
                    self.blink_history.clear()
                    return "Left Click"
        return None

    @torch.inference_mode()
    def map_to_screen(self, x, y):
        """Map coordinates to screen using PyTorch"""
        if not self.calibrated:
            return self.screen_width // 2, self.screen_height // 2
        
        x_tensor = torch.tensor([x], device=self.device)
        y_tensor = torch.tensor([y], device=self.device)
        
        # Linear interpolation using PyTorch
        screen_x = torch.lerp(torch.tensor([0.], device=self.device),
                            torch.tensor([float(self.screen_width)], device=self.device),
                            (x_tensor - self.cal_left) / (self.cal_right - self.cal_left))
        
        screen_y = torch.lerp(torch.tensor([0.], device=self.device),
                            torch.tensor([float(self.screen_height)], device=self.device),
                            (y_tensor - self.cal_top) / (self.cal_bottom - self.cal_top))
        
        return screen_x.item(), screen_y.item()

    @torch.inference_mode()
    def smooth_movement(self, current_x, current_y):
        """Apply smoothing using PyTorch"""
        # Convert current and previous positions to float tensors
        current_pos = torch.tensor([current_x, current_y], device=self.device, dtype=torch.float32)
        prev_pos = torch.tensor([self.prev_x.item(), self.prev_y.item()], device=self.device, dtype=torch.float32)
        
        # Perform linear interpolation
        smoothed = torch.lerp(prev_pos, current_pos, self.smooth_factor)
        
        # Update previous positions
        self.prev_x = smoothed[0]
        self.prev_y = smoothed[1]
        
        return int(smoothed[0].item()), int(smoothed[1].item())

    def run(self):
        """Main loop with PyTorch acceleration"""
        self.calibrate()
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            # Convert frame to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            results = self.face_mesh.process(rgb_frame)
            
            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                
                if self.detect_blink(face_landmarks):
                    current_time = time.time()
                    if current_time - self.last_blink_time > 0.1:
                        self.blink_history.append(current_time)
                        self.last_blink_time = current_time
                
                click_status = self.process_blinks()
                
                # Nose tracking
                nose_tip = face_landmarks.landmark[self.NOSE_TIP]
                nose_x = nose_tip.x * frame.shape[1]
                nose_y = nose_tip.y * frame.shape[0]
                
                screen_x, screen_y = self.map_to_screen(nose_x, nose_y)
                smooth_x, smooth_y = self.smooth_movement(screen_x, screen_y)
                
                pyautogui.moveTo(smooth_x, smooth_y)
                
                # Visual feedback
                cv2.circle(frame, (int(nose_x), int(nose_y)), 5, (0, 255, 0), -1)
                
                if click_status:
                    cv2.putText(frame, click_status, (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                cv2.putText(frame, f"Blink count: {len(self.blink_history)}", 
                          (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                if self.calibrated:
                    cv2.rectangle(frame, 
                                (int(self.cal_left), int(self.cal_top)),
                                (int(self.cal_right), int(self.cal_bottom)),
                                (255, 0, 0), 2)
            
            cv2.imshow('Nose Tracking', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def calibrate(self):
        """Calibration process"""
        print("Starting calibration...")
        calibration_points = []
        start_time = time.time()
        calibration_duration = 15
        
        while time.time() - start_time < calibration_duration:
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)
            
            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                nose_tip = face_landmarks.landmark[self.NOSE_TIP]
                frame_height, frame_width = frame.shape[:2]
                nose_x = nose_tip.x * frame_width
                nose_y = nose_tip.y * frame_height
                
                calibration_points.append((nose_x, nose_y))
                cv2.circle(frame, (int(nose_x), int(nose_y)), 5, (0, 255, 0), -1)
            
            # Display countdown and instructions
            remaining_time = int(calibration_duration - (time.time() - start_time))
            cv2.putText(frame, "Move head around to define your range of motion", 
                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Time remaining: {remaining_time} seconds", 
                      (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow('Calibration', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cv2.destroyAllWindows()
        
        if calibration_points:
            calibration_points = np.array(calibration_points)
            self.cal_left = calibration_points[:, 0].min()
            self.cal_right = calibration_points[:, 0].max()
            self.cal_top = calibration_points[:, 1].min()
            self.cal_bottom = calibration_points[:, 1].max()
            self.calibrated = True

        print("Calibration complete!")

    def __del__(self):
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    tracker = TorchNoseTracker()
    tracker.run()
