import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time
import math
import os
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from screeninfo import get_monitors
import pygetwindow as gw

# --- CONFIGURATION ---

# Configuration des points centraux de chaque écran (Calibration utilisateur)
SCREEN_CENTERS = {
    "ECRAN_1_GAUCHE": (-5, 6),   # Ecran gauche
    "ECRAN_2_CENTRE": (6, 10),   # Ecran central
    "ECRAN_3_HAUT":   (-1, 10),  # Ecran haut
    "ECRAN_4_DROITE": (11, 7)    # Ecran droite
}

SMOOTHING_FACTOR = 0.5
SCROLL_SPEED = 20
VOLUME_DEBOUNCE = 0.1 # Vitesse de répétition du volume

class EyeGestureController:
    def __init__(self):
        # Chemins des modèles
        cwd = os.path.dirname(os.path.abspath(__file__))
        face_model_path = os.path.join(cwd, 'face_landmarker.task')
        hand_model_path = os.path.join(cwd, 'hand_landmarker.task')

        # Initialisation Face Landmarker
        base_options_face = python.BaseOptions(model_asset_path=face_model_path)
        options_face = vision.FaceLandmarkerOptions(
            base_options=base_options_face,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=1)
        self.face_landmarker = vision.FaceLandmarker.create_from_options(options_face)

        # Initialisation Hand Landmarker
        base_options_hand = python.BaseOptions(model_asset_path=hand_model_path)
        options_hand = vision.HandLandmarkerOptions(
            base_options=base_options_hand,
            num_hands=1)
        self.hand_landmarker = vision.HandLandmarker.create_from_options(options_hand)
        
        self.cap = cv2.VideoCapture(0)
        
        # Gestion multi-écrans
        self.monitors = get_monitors()
        self.monitor_mapping = {
            "ECRAN_1_GAUCHE": 0,
            "ECRAN_2_CENTRE": 3,
            "ECRAN_3_HAUT":   2,
            "ECRAN_4_DROITE": 1
        }
        
        # État
        self.current_screen = "INCONNU"
        self.active_monitor = self.monitors[0] if self.monitors else None
        self.prev_mouse_x, self.prev_mouse_y = pyautogui.position()
        self.last_volume_time = 0

    def update_active_monitor(self, screen_name):
        if screen_name in self.monitor_mapping:
            idx = self.monitor_mapping[screen_name]
            if idx < len(self.monitors):
                self.active_monitor = self.monitors[idx]

    def draw_debug_window(self):
        # Création d'une image noire 400x300
        debug_img = np.zeros((300, 400, 3), dtype=np.uint8)
        
        # Définition des rectangles (Schéma approximatif de votre setup)
        # (x, y, w, h)
        screens = {
            "ECRAN_3_HAUT":   (100, 20,  200, 100), # Haut Centre
            "ECRAN_1_GAUCHE": (20,  130, 120, 100), # Gauche Bas
            "ECRAN_2_CENTRE": (150, 130, 120, 100), # Centre Bas
            "ECRAN_4_DROITE": (280, 50,  100, 180)  # Droite Vertical
        }
        
        for name, rect in screens.items():
            x, y, w, h = rect
            color = (100, 100, 100) # Gris par défaut
            thickness = 2
            
            if name == self.current_screen:
                color = (0, 0, 255) # Rouge si actif
                thickness = -1 # Rempli
            
            cv2.rectangle(debug_img, (x, y), (x+w, y+h), color, thickness)
            cv2.putText(debug_img, name.split('_')[2], (x+5, y+h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        cv2.imshow("Radar Ecrans", debug_img)

    def get_head_pose(self, image_shape, landmarks):
        img_h, img_w, _ = image_shape
        face_3d = []
        face_2d = []
        key_points = [1, 199, 33, 263, 61, 291]

        for idx in key_points:
            lm = landmarks[idx]
            x, y = int(lm.x * img_w), int(lm.y * img_h)
            face_2d.append([x, y])
            face_3d.append([x, y, lm.z])

        face_2d = np.array(face_2d, dtype=np.float64)
        face_3d = np.array(face_3d, dtype=np.float64)
        focal_length = 1 * img_w
        cam_matrix = np.array([[focal_length, 0, img_h / 2], [0, focal_length, img_w / 2], [0, 0, 1]])
        dist_matrix = np.zeros((4, 1), dtype=np.float64)

        success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
        rmat, jac = cv2.Rodrigues(rot_vec)
        angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

        return angles[1] * 360, angles[0] * 360 # Yaw, Pitch

    def determine_screen(self, yaw, pitch):
        best_screen = "INCONNU"
        min_dist = float('inf')
        for screen_name, center in SCREEN_CENTERS.items():
            c_yaw, c_pitch = center
            dist = math.sqrt((yaw - c_yaw)**2 + (pitch - c_pitch)**2)
            if dist < min_dist:
                min_dist = dist
                best_screen = screen_name
        return best_screen

    def detect_hand_gesture(self, landmarks):
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        ring_tip = landmarks[16]
        pinky_tip = landmarks[20]
        wrist = landmarks[0]
        
        # Distance Pouce-Index (Zoom/Scroll)
        pinch_dist = math.hypot(index_tip.x - thumb_tip.x, index_tip.y - thumb_tip.y)
        
        # Doigts levés ?
        index_up = index_tip.y < landmarks[6].y
        middle_up = middle_tip.y < landmarks[10].y
        ring_up = ring_tip.y < landmarks[14].y
        pinky_up = pinky_tip.y < landmarks[18].y
        
        # Calcul du Roll (Inclinaison de la main)
        # Vecteur Poignet -> Majeur
        dx = middle_tip.x - wrist.x
        dy = middle_tip.y - wrist.y
        # Angle en degrés (0 = vertical)
        roll_angle = math.degrees(math.atan2(dx, -dy))
        
        if pinch_dist < 0.05:
            return "PINCH", index_tip, roll_angle
        
        if index_up and not middle_up and not ring_up and not pinky_up:
            return "POINT", index_tip, roll_angle

        if index_up and middle_up and not ring_up and not pinky_up:
            return "TWO_FINGERS", index_tip, roll_angle
            
        if not index_up and not middle_up and not ring_up and not pinky_up:
            return "FIST", None, roll_angle
            
        # Main ouverte (tous les doigts levés)
        if index_up and middle_up and ring_up and pinky_up:
            return "OPEN_PALM", None, roll_angle
            
        return "NONE", None, roll_angle

    def control_mouse(self, gesture, position, roll_angle):
        if not self.active_monitor:
            return

        # --- ECRAN 3 (Films) ---
        if self.current_screen == "ECRAN_3_HAUT":
            # PAUSE
            if gesture == "FIST":
                current_time = time.time()
                if not hasattr(self, 'last_pause_time'): self.last_pause_time = 0
                if current_time - self.last_pause_time > 1.0:
                    try:
                        center_x = self.active_monitor.x + self.active_monitor.width // 2
                        center_y = self.active_monitor.y + self.active_monitor.height // 2
                        windows = gw.getWindowsAt(center_x, center_y)
                        target_window = None
                        if windows:
                            for w in windows:
                                if "Eye & Gesture Control" not in w.title:
                                    target_window = w
                                    break
                        if target_window:
                            try:
                                if target_window.isMinimized: target_window.restore()
                                target_window.activate()
                            except: pass
                            time.sleep(0.1)
                            pyautogui.press('space')
                        else:
                            pyautogui.click(center_x, center_y)
                            time.sleep(0.1)
                            pyautogui.press('space')
                    except Exception as e:
                        print(f"Erreur: {e}")
                        pyautogui.press('space')
                    self.last_pause_time = current_time
            
            # VOLUME (Main Ouverte + Rotation)
            elif gesture == "OPEN_PALM":
                current_time = time.time()
                if current_time - self.last_volume_time > VOLUME_DEBOUNCE:
                    if roll_angle > 20: # Penché à droite
                        pyautogui.press('volumeup')
                        print(f"Volume UP (Angle: {int(roll_angle)})")
                        self.last_volume_time = current_time
                    elif roll_angle < -20: # Penché à gauche
                        pyautogui.press('volumedown')
                        print(f"Volume DOWN (Angle: {int(roll_angle)})")
                        self.last_volume_time = current_time
            return

        # --- ECRAN 1 (Web) ---
        if self.current_screen == "ECRAN_1_GAUCHE":
            if gesture == "POINT" and position:
                screen_x = np.interp(position.x, [0, 1], [0, self.active_monitor.width])
                screen_y = np.interp(position.y, [0, 1], [0, self.active_monitor.height])
                target_x = self.active_monitor.x + screen_x
                target_y = self.active_monitor.y + screen_y
                curr_x = self.prev_mouse_x + (target_x - self.prev_mouse_x) * SMOOTHING_FACTOR
                curr_y = self.prev_mouse_y + (target_y - self.prev_mouse_y) * SMOOTHING_FACTOR
                pyautogui.moveTo(curr_x, curr_y)
                self.prev_mouse_x, self.prev_mouse_y = curr_x, curr_y
            elif gesture == "TWO_FINGERS":
                pyautogui.scroll(SCROLL_SPEED)
            elif gesture == "PINCH":
                pyautogui.scroll(-SCROLL_SPEED)
            return

        # --- DEFAUT ---
        if gesture == "POINT" and position:
            screen_x = np.interp(position.x, [0, 1], [0, self.active_monitor.width])
            screen_y = np.interp(position.y, [0, 1], [0, self.active_monitor.height])
            target_x = self.active_monitor.x + screen_x
            target_y = self.active_monitor.y + screen_y
            curr_x = self.prev_mouse_x + (target_x - self.prev_mouse_x) * SMOOTHING_FACTOR
            curr_y = self.prev_mouse_y + (target_y - self.prev_mouse_y) * SMOOTHING_FACTOR
            pyautogui.moveTo(curr_x, curr_y)
            self.prev_mouse_x, self.prev_mouse_y = curr_x, curr_y
        elif gesture == "PINCH" and position:
            if position.y < 0.4: pyautogui.scroll(SCROLL_SPEED)
            elif position.y > 0.6: pyautogui.scroll(-SCROLL_SPEED)

    def run(self):
        print("Démarrage... Appuyez sur 'Esc' pour quitter.")
        while self.cap.isOpened():
            success, image = self.cap.read()
            if not success: continue

            image = cv2.flip(image, 1)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            
            # Head Tracking
            face_result = self.face_landmarker.detect(mp_image)
            if face_result.face_landmarks:
                landmarks = face_result.face_landmarks[0]
                yaw, pitch = self.get_head_pose(image.shape, landmarks)
                new_screen = self.determine_screen(yaw, pitch)
                
                if new_screen != self.current_screen:
                    self.current_screen = new_screen
                    self.update_active_monitor(new_screen)
                
                cv2.putText(image, f"Screen: {self.current_screen}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Hand Tracking
            hand_result = self.hand_landmarker.detect(mp_image)
            if hand_result.hand_landmarks:
                for landmarks in hand_result.hand_landmarks:
                    for lm in landmarks:
                        x, y = int(lm.x * image.shape[1]), int(lm.y * image.shape[0])
                        cv2.circle(image, (x, y), 5, (255, 0, 0), -1)
                    
                    gesture, pos, roll = self.detect_hand_gesture(landmarks)
                    cv2.putText(image, f"Gesture: {gesture} Roll: {int(roll)}", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                    self.control_mouse(gesture, pos, roll)

            # Visual Debug
            self.draw_debug_window()

            cv2.imshow('Eye & Gesture Control', image)
            if cv2.waitKey(5) & 0xFF == 27: break
                
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    controller = EyeGestureController()
    controller.run()
