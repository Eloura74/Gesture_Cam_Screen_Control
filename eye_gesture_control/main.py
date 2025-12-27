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
VOLUME_DEBOUNCE = 0.1
KNOB_SENSITIVITY = 5 # Plus sensible (Pixels de scroll par degré)

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
        
        # État Molette Virtuelle
        self.knob_active = False
        self.knob_ref_angle = 0
        self.knob_accumulated_scroll = 0

    def update_active_monitor(self, screen_name):
        if screen_name in self.monitor_mapping:
            idx = self.monitor_mapping[screen_name]
            if idx < len(self.monitors):
                self.active_monitor = self.monitors[idx]

    def draw_debug_window(self):
        debug_img = np.zeros((300, 400, 3), dtype=np.uint8)
        screens = {
            "ECRAN_3_HAUT":   (100, 20,  200, 100),
            "ECRAN_1_GAUCHE": (20,  130, 120, 100),
            "ECRAN_2_CENTRE": (150, 130, 120, 100),
            "ECRAN_4_DROITE": (280, 50,  100, 180)
        }
        
        for name, rect in screens.items():
            x, y, w, h = rect
            color = (100, 100, 100)
            thickness = 2
            if name == self.current_screen:
                color = (0, 0, 255)
                thickness = -1
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
        
        # Helper pour savoir si un doigt est étendu (Distance Bout-Poignet > Distance Jointure-Poignet)
        def is_extended(tip, joint):
            return math.hypot(tip.x - wrist.x, tip.y - wrist.y) > math.hypot(joint.x - wrist.x, joint.y - wrist.y)

        # On utilise PIP (6, 10, 14, 18) pour les doigts et IP (3) pour le pouce
        thumb_extended = is_extended(thumb_tip, landmarks[2]) # Comparé au MCP pour le pouce
        index_extended = is_extended(index_tip, landmarks[6])
        middle_extended = is_extended(middle_tip, landmarks[10])
        ring_extended = is_extended(ring_tip, landmarks[14])
        pinky_extended = is_extended(pinky_tip, landmarks[18])

        # Distance Pouce-Index (Zoom/Scroll)
        pinch_dist = math.hypot(index_tip.x - thumb_tip.x, index_tip.y - thumb_tip.y)
        
        # Calcul du Roll (Inclinaison de la main)
        dx = middle_tip.x - wrist.x
        dy = middle_tip.y - wrist.y
        roll_angle = math.degrees(math.atan2(dx, -dy))
        
        # PINCH (Prioritaire)
        if pinch_dist < 0.05:
            return "PINCH", index_tip, roll_angle
        
        # NOUVEAU : THREE_FINGERS (Pouce + Index + Majeur levés) -> Molette
        if thumb_extended and index_extended and middle_extended and not ring_extended and not pinky_extended:
            return "THREE_FINGERS", index_tip, roll_angle

        # POINT (Index seul)
        if index_extended and not middle_extended and not ring_extended and not pinky_extended:
            return "POINT", index_tip, roll_angle

        # TWO_FINGERS (Index + Majeur)
        if index_extended and middle_extended and not ring_extended and not pinky_extended:
            return "TWO_FINGERS", index_tip, roll_angle
            
        # FIST (Poing)
        if not index_extended and not middle_extended and not ring_extended and not pinky_extended:
            return "FIST", None, roll_angle
            
        # OPEN_PALM (Main ouverte)
        if index_extended and middle_extended and ring_extended and pinky_extended:
            return "OPEN_PALM", None, roll_angle
            
        return "NONE", None, roll_angle

    def control_mouse(self, gesture, position, roll_angle, image=None):
        if not self.active_monitor:
            return

        # --- ECRAN 3 (Films) ---
        if self.current_screen == "ECRAN_3_HAUT":
            self.knob_active = False # Reset knob
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
                        pyautogui.press('space')
                    self.last_pause_time = current_time
            
            elif gesture == "OPEN_PALM":
                current_time = time.time()
                if current_time - self.last_volume_time > VOLUME_DEBOUNCE:
                    if roll_angle > 20:
                        pyautogui.press('volumeup')
                        self.last_volume_time = current_time
                    elif roll_angle < -20:
                        pyautogui.press('volumedown')
                        self.last_volume_time = current_time
            return

        # --- ECRAN 1 (Web) ---
        if self.current_screen == "ECRAN_1_GAUCHE":
            if gesture == "THREE_FINGERS" and position:
                # MODE JOYSTICK (Pouce + Index + Majeur)
                # Inclinaison maintenue = Scroll continu
                
                # Seuil d'activation (en degrés)
                DEADZONE = 15
                
                # Scroll Bas (Rotation Droite)
                if roll_angle > DEADZONE:
                    # Vitesse proportionnelle à l'angle
                    intensity = (roll_angle - DEADZONE) / 5 # Facteur d'accélération
                    speed = int(SCROLL_SPEED * intensity)
                    pyautogui.scroll(-speed)
                    
                # Scroll Haut (Rotation Gauche)
                elif roll_angle < -DEADZONE:
                    intensity = (abs(roll_angle) - DEADZONE) / 5
                    speed = int(SCROLL_SPEED * intensity)
                    pyautogui.scroll(speed)
                
                # DESSIN UI
                if image is not None:
                    h, w, _ = image.shape
                    cx, cy = int(position.x * w), int(position.y * h)
                    
                    # Couleur change selon l'activation
                    color = (0, 255, 255) # Jaune (Neutre)
                    if abs(roll_angle) > DEADZONE:
                        color = (0, 255, 0) # Vert (Actif)
                        
                    # Cercle extérieur
                    cv2.circle(image, (cx, cy), 45, color, 2)
                    
                    # Indicateur de rotation (Aiguille)
                    rad = math.radians(roll_angle - 90)
                    ex = int(cx + 45 * math.cos(rad))
                    ey = int(cy + 45 * math.sin(rad))
                    cv2.line(image, (cx, cy), (ex, ey), color, 2)
                    
                    # Zones mortes (visuel)
                    rad_p = math.radians(DEADZONE - 90)
                    rad_m = math.radians(-DEADZONE - 90)
                    ex_p = int(cx + 35 * math.cos(rad_p))
                    ey_p = int(cy + 35 * math.sin(rad_p))
                    ex_m = int(cx + 35 * math.cos(rad_m))
                    ey_m = int(cy + 35 * math.sin(rad_m))
                    cv2.line(image, (cx, cy), (ex_p, ey_p), (100, 100, 100), 1)
                    cv2.line(image, (cx, cy), (ex_m, ey_m), (100, 100, 100), 1)
                    
                    cv2.putText(image, "SCROLL", (cx-30, cy-55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            else:
                self.knob_active = False
                
                if gesture == "POINT" and position:
                    screen_x = np.interp(position.x, [0, 1], [0, self.active_monitor.width])
                    screen_y = np.interp(position.y, [0, 1], [0, self.active_monitor.height])
                    target_x = self.active_monitor.x + screen_x
                    target_y = self.active_monitor.y + screen_y
                    curr_x = self.prev_mouse_x + (target_x - self.prev_mouse_x) * SMOOTHING_FACTOR
                    curr_y = self.prev_mouse_y + (target_y - self.prev_mouse_y) * SMOOTHING_FACTOR
                    pyautogui.moveTo(curr_x, curr_y)
                    self.prev_mouse_x, self.prev_mouse_y = curr_x, curr_y
            return

        # --- DEFAUT ---
        self.knob_active = False
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
                    gesture, pos, roll = self.detect_hand_gesture(landmarks)
                    
                    # Dessin des points (sauf si Knob actif pour clarté)
                    if not (self.current_screen == "ECRAN_1_GAUCHE" and gesture == "THREE_FINGERS"):
                        for lm in landmarks:
                            x, y = int(lm.x * image.shape[1]), int(lm.y * image.shape[0])
                            cv2.circle(image, (x, y), 5, (255, 0, 0), -1)
                    
                    cv2.putText(image, f"Gesture: {gesture}", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                    
                    # On passe l'image pour dessiner le Knob
                    self.control_mouse(gesture, pos, roll, image)

            # Visual Debug
            self.draw_debug_window()

            cv2.imshow('Eye & Gesture Control', image)
            if cv2.waitKey(5) & 0xFF == 27: break
                
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    controller = EyeGestureController()
    controller.run()
