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

SCREEN_CENTERS = {
    "ECRAN_1_GAUCHE": (-5, 6),   # Ecran gauche
    "ECRAN_2_CENTRE": (6, 10),   # Ecran central
    "ECRAN_3_HAUT":   (-1, 10),  # Ecran haut
    "ECRAN_4_DROITE": (11, 7)    # Ecran droite
}

SMOOTHING_FACTOR = 0.5
SCROLL_SPEED = 20
VOLUME_DEBOUNCE = 0.1
KNOB_SENSITIVITY = 5
AFK_TIMEOUT = 2.0
SWIPE_THRESHOLD = 0.04 # Encore plus sensible
SWIPE_TIME_WINDOW = 0.6 # Plus de temps pour le geste
SWIPE_COOLDOWN = 1.5

# --- CLASSE ---
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
        
        # État Smart Pause
        self.last_face_time = time.time()
        self.is_paused_afk = False
        self.last_pause_action_time = 0
        
        # État Swipe
        self.hand_history = [] # [(time, x), ...]
        self.last_swipe_time = 0
        self.last_swipe_direction = None
        self.last_swipe_display_time = 0
        
        # État Onglets (Opera)
        self.current_tab = 1

        # État Onglets (Opera)
        self.current_tab = 1

    # --- Fonction pour la gestion des écrans ---
    def update_active_monitor(self, screen_name):
        if screen_name in self.monitor_mapping:
            idx = self.monitor_mapping[screen_name]
            if idx < len(self.monitors):
                self.active_monitor = self.monitors[idx]

    # --- Fonction pour le debug (RADAR UI 3.0 - Realistic) ---
    def draw_debug_window(self):
        # Fond gris foncé (Mur)
        debug_img = np.full((350, 500, 3), 30, dtype=np.uint8)
        
        # Couleurs
        COLOR_BEZEL = (20, 20, 20)
        COLOR_SCREEN_OFF = (10, 10, 10)
        COLOR_SCREEN_ACTIVE = (0, 100, 0) # Vert sombre
        COLOR_STAND = (50, 50, 50)
        COLOR_TEXT = (200, 200, 200)
        COLOR_HIGHLIGHT = (0, 255, 0) # Vert néon
        
        # Configuration Physique (x, y, w, h)
        # Centre = (200, 150)
        screens = {
            "ECRAN_3_HAUT":   (180, 40,  140, 80),   # Haut
            "ECRAN_1_GAUCHE": (30,  130, 140, 80),   # Gauche
            "ECRAN_2_CENTRE": (180, 130, 140, 80),   # Centre
            "ECRAN_4_DROITE": (330, 80,  60,  180)   # Droite (Vertical)
        }
        
        # Dessin des pieds (Stands)
        # Pied Centre
        cv2.rectangle(debug_img, (240, 210), (260, 250), COLOR_STAND, -1)
        cv2.rectangle(debug_img, (220, 250), (280, 260), COLOR_STAND, -1)
        # Pied Gauche
        cv2.rectangle(debug_img, (90, 210), (110, 250), COLOR_STAND, -1)
        cv2.rectangle(debug_img, (70, 250), (130, 260), COLOR_STAND, -1)
        # Pied Droite (Bras articulé supposé ou pied)
        cv2.rectangle(debug_img, (350, 260), (370, 270), COLOR_STAND, -1)

        # Dessin des Ecrans
        for name, rect in screens.items():
            x, y, w, h = rect
            
            # Bezel (Cadre)
            cv2.rectangle(debug_img, (x-5, y-5), (x+w+5, y+h+5), COLOR_BEZEL, -1)
            
            # Ecran (Dalle)
            if name == self.current_screen:
                # Ecran Actif : Effet "Matrix" ou Vert
                cv2.rectangle(debug_img, (x, y), (x+w, y+h), COLOR_SCREEN_ACTIVE, -1)
                cv2.rectangle(debug_img, (x, y), (x+w, y+h), COLOR_HIGHLIGHT, 2)
            else:
                # Ecran Inactif
                cv2.rectangle(debug_img, (x, y), (x+w, y+h), COLOR_SCREEN_OFF, -1)
                cv2.rectangle(debug_img, (x, y), (x+w, y+h), (60, 60, 60), 1)

            # Nom
            short_name = name.split('_')[2]
            text_size = cv2.getTextSize(short_name, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
            text_x = x + (w - text_size[0]) // 2
            text_y = y + (h + text_size[1]) // 2
            cv2.putText(debug_img, short_name, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_TEXT, 1)

        # Bureau (Table)
        cv2.rectangle(debug_img, (0, 260), (500, 350), (40, 40, 40), -1)

        # Status Overlay
        if self.is_paused_afk:
            overlay = debug_img.copy()
            cv2.rectangle(overlay, (0, 0), (500, 350), (0, 0, 50), -1)
            cv2.addWeighted(overlay, 0.5, debug_img, 0.5, 0, debug_img)
            cv2.putText(debug_img, "AFK PAUSE", (180, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        elif time.time() - self.last_swipe_display_time < 1.0:
             cv2.putText(debug_img, f"SWIPE {self.last_swipe_direction} >>", (150, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.imshow("Radar Ecrans", debug_img)

    # --- Fonction pour la détection de la position de la tête ---
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

    # --- Fonction pour la détection de l'écran ---
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
    
    # --- Fonction pour la détection des gestes de la main ---
    def detect_hand_gesture(self, landmarks):
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        ring_tip = landmarks[16]
        pinky_tip = landmarks[20]
        wrist = landmarks[0]
        
        def is_extended(tip, joint):
            return math.hypot(tip.x - wrist.x, tip.y - wrist.y) > math.hypot(joint.x - wrist.x, joint.y - wrist.y)

        thumb_extended = is_extended(thumb_tip, landmarks[2]) 
        index_extended = is_extended(index_tip, landmarks[6]) 
        middle_extended = is_extended(middle_tip, landmarks[10]) 
        ring_extended = is_extended(ring_tip, landmarks[14]) 
        pinky_extended = is_extended(pinky_tip, landmarks[18]) 

        # Correction Pouce : Si le bout du pouce est proche de la base de l'auriculaire, il est fermé
        # (Distance normalisée)
        pinky_mcp = landmarks[17]
        dist_thumb_pinky = math.hypot(thumb_tip.x - pinky_mcp.x, thumb_tip.y - pinky_mcp.y)
        if dist_thumb_pinky < 0.15:
            thumb_extended = False

        pinch_dist = math.hypot(index_tip.x - thumb_tip.x, index_tip.y - thumb_tip.y)
        
        dx = middle_tip.x - wrist.x 
        dy = middle_tip.y - wrist.y 
        roll_angle = math.degrees(math.atan2(dx, -dy)) 
        
        if pinch_dist < 0.05:
            return "PINCH", index_tip, roll_angle
        
        if thumb_extended and index_extended and middle_extended and not ring_extended and not pinky_extended:
            return "THREE_FINGERS", index_tip, roll_angle

        if index_extended and not middle_extended and not ring_extended and not pinky_extended:
            return "POINT", index_tip, roll_angle

        if index_extended and middle_extended and not ring_extended and not pinky_extended:
            return "TWO_FINGERS", index_tip, roll_angle
            
        if not index_extended and not middle_extended and not ring_extended and not pinky_extended:
            return "FIST", None, roll_angle
            
        if index_extended and middle_extended and ring_extended and pinky_extended:
            return "OPEN_PALM", None, roll_angle
            
        return "NONE", None, roll_angle

    # --- Fonction pour détecter le Swipe ---
    def detect_swipe(self, position):
        current_time = time.time()
        # Ajouter la position actuelle (x)
        self.hand_history.append((current_time, position.x))
        
        # Nettoyer l'historique (garder seulement les dernières X secondes)
        self.hand_history = [h for h in self.hand_history if current_time - h[0] < SWIPE_TIME_WINDOW]
        
        if len(self.hand_history) < 2:
            return None
            
        if current_time - self.last_swipe_time < SWIPE_COOLDOWN:
            return None

        # Calculer le déplacement
        start_x = self.hand_history[0][1]
        end_x = self.hand_history[-1][1]
        dx = end_x - start_x
        
        # Debug Swipe
        # if abs(dx) > 0.02:
        #     print(f"Swipe Delta: {dx:.3f}")
        
        # Seuil abaissé pour plus de facilité
        if abs(dx) > SWIPE_THRESHOLD:
            self.last_swipe_time = current_time
            self.hand_history = [] # Reset
            direction = "RIGHT" if dx > 0 else "LEFT"
            self.last_swipe_direction = direction
            self.last_swipe_display_time = current_time
            return direction
            
        return None

    # --- Fonction pour le play pause ---
    def trigger_play_pause(self):
        """Active la fenêtre sur l'écran 3 et appuie sur Espace"""
        monitor_idx = self.monitor_mapping.get("ECRAN_3_HAUT")
        if monitor_idx is None or monitor_idx >= len(self.monitors):
            return

        monitor = self.monitors[monitor_idx]
        center_x = monitor.x + monitor.width // 2
        center_y = monitor.y + monitor.height // 2

        try:
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
                print("SMART PAUSE/PLAY (Window)")
            else:
                pyautogui.click(center_x, center_y)
                time.sleep(0.1)
                pyautogui.press('space')
                print("SMART PAUSE/PLAY (Fallback Click)")
        except Exception as e:
            print(f"Erreur Smart Pause: {e}")
            pyautogui.press('space')

    def control_mouse(self, gesture, position, roll_angle, image=None):
        # print(f"Control Mouse: {gesture} on {self.current_screen}") # Debug
        if not self.active_monitor:
            return

        # --- ECRAN 3 (Films) ---
        if self.current_screen == "ECRAN_3_HAUT":
            self.knob_active = False 
            if gesture == "FIST":
                current_time = time.time()
                if current_time - self.last_pause_action_time > 1.0:
                    self.trigger_play_pause()
                    self.last_pause_action_time = current_time
            
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
            # DEBUG TYPING (Poing)
            if gesture == "FIST":
                # print("DEBUG: TYPING 'fist_ok'", flush=True)
                # pyautogui.write("fist_ok")
                time.sleep(1.0) # Debounce
                return

            # SWIPE (Index + Majeur uniquement)
            # Condition : Geste "Pistolet" (Main horizontale)
            is_horizontal = abs(roll_angle) > 30
            
            if gesture == "TWO_FINGERS" and position and is_horizontal:
                swipe = self.detect_swipe(position)
                if swipe:
                    # Mapping AZERTY pour les chiffres (1=&, 2=é, 3=", etc.)
                    AZERTY_KEYS = ['&', 'é', '"', "'", '(', '-', 'è', '_']
                    
                    if swipe == "RIGHT":
                        self.current_tab = min(8, self.current_tab + 1)
                    elif swipe == "LEFT":
                        self.current_tab = max(1, self.current_tab - 1)
                    
                    print(f"EXECUTING SWIPE: {swipe} -> Tab {self.current_tab}", flush=True)
                    
                    # Debug Focus
                    try:
                        active_win = gw.getActiveWindow()
                        if active_win:
                            print(f"Active Window: {active_win.title}", flush=True)
                    except:
                        pass

                    target_char = str(self.current_tab)
                    azerty_char = AZERTY_KEYS[self.current_tab - 1]
                    numpad_key = f"num{self.current_tab}"

                    # print(f"INJECTING (Explicit): Ctrl + {azerty_char} ...", flush=True)

                    # 1. Tentative Explicite (Ctrl + Touche AZERTY)
                    pyautogui.keyDown('ctrl')
                    time.sleep(0.1)
                    pyautogui.press(azerty_char)
                    time.sleep(0.1)
                    pyautogui.keyUp('ctrl')
                    
                    # 2. FALLBACK : Si ça n'a pas marché, on force le défilement relatif
                    # (Au moins ça bouge !)
                    print(f"INJECTING (Fallback): Ctrl + PageUp/Down", flush=True)
                    pyautogui.keyDown('ctrl')
                    if swipe == "RIGHT":
                        pyautogui.press('pagedown') # Onglet Suivant
                    else:
                        pyautogui.press('pageup')   # Onglet Précédent
                    pyautogui.keyUp('ctrl')
                    
                    return

            # JOYSTICK SCROLL (3 Doigts)

            # JOYSTICK SCROLL (3 Doigts)
            if gesture == "THREE_FINGERS" and position:
                DEADZONE = 15
                if roll_angle > DEADZONE:
                    intensity = (roll_angle - DEADZONE) / 5
                    speed = int(SCROLL_SPEED * intensity)
                    pyautogui.scroll(-speed)
                elif roll_angle < -DEADZONE:
                    intensity = (abs(roll_angle) - DEADZONE) / 5
                    speed = int(SCROLL_SPEED * intensity)
                    pyautogui.scroll(speed)
                
                if image is not None:
                    h, w, _ = image.shape
                    cx, cy = int(position.x * w), int(position.y * h)
                    color = (0, 255, 255)
                    if abs(roll_angle) > DEADZONE: color = (0, 255, 0)
                    cv2.circle(image, (cx, cy), 45, color, 2)
                    rad = math.radians(roll_angle - 90)
                    ex = int(cx + 45 * math.cos(rad))
                    ey = int(cy + 45 * math.sin(rad))
                    cv2.line(image, (cx, cy), (ex, ey), color, 2)
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

    # --- Fonction pour le run ---
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
            current_time = time.time()
            
            if face_result.face_landmarks:
                self.last_face_time = current_time
                # Si on était en pause AFK, on reprend
                if self.is_paused_afk:
                    print("Visage retrouvé : RESUME")
                    self.trigger_play_pause()
                    self.is_paused_afk = False
                
                landmarks = face_result.face_landmarks[0]
                yaw, pitch = self.get_head_pose(image.shape, landmarks)
                new_screen = self.determine_screen(yaw, pitch)
                
                if new_screen != self.current_screen:
                    self.current_screen = new_screen
                    self.update_active_monitor(new_screen)
                
                cv2.putText(image, f"Screen: {self.current_screen}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                # Pas de visage
                if not self.is_paused_afk and (current_time - self.last_face_time > AFK_TIMEOUT):
                    print("AFK détecté : PAUSE")
                    self.trigger_play_pause()
                    self.is_paused_afk = True

            # Hand Tracking
            hand_result = self.hand_landmarker.detect(mp_image)
            if hand_result.hand_landmarks:
                for i, landmarks in enumerate(hand_result.hand_landmarks):
                    # FILTRE MAIN DROITE (Désactivé pour debug)
                    handedness = hand_result.handedness[i][0]
                    
                    # if handedness.category_name == "Left":
                    if True: # On accepte tout pour tester
                        gesture, pos, roll = self.detect_hand_gesture(landmarks)
                        # print(f"Hand: {handedness.category_name}, Gesture: {gesture}, Screen: {self.current_screen}", flush=True)
                        
                        # Dessin des points (Style Cyber Skeleton)
                        # if not (self.current_screen == "ECRAN_1_GAUCHE" and gesture == "THREE_FINGERS"):
                        if True:
                            h, w, _ = image.shape
                            # Connexions des doigts
                            connections = [
                                (0, 1), (1, 2), (2, 3), (3, 4),         # Pouce
                                (0, 5), (5, 6), (6, 7), (7, 8),         # Index
                                (0, 9), (9, 10), (10, 11), (11, 12),    # Majeur
                                (0, 13), (13, 14), (14, 15), (15, 16),  # Annulaire
                                (0, 17), (17, 18), (18, 19), (19, 20),  # Auriculaire
                                (5, 9), (9, 13), (13, 17)               # Paume
                            ]
                            
                            # Dessin des os (Lignes)
                            for start_idx, end_idx in connections:
                                start = landmarks[start_idx]
                                end = landmarks[end_idx]
                                start_point = (int(start.x * w), int(start.y * h))
                                end_point = (int(end.x * w), int(end.y * h))
                                cv2.line(image, start_point, end_point, (255, 255, 0), 2) # Cyan

                            # Dessin des jointures (Cercles)
                            for lm in landmarks:
                                x, y = int(lm.x * w), int(lm.y * h)
                                cv2.circle(image, (x, y), 4, (255, 0, 255), -1) # Magenta
                        
                        cv2.putText(image, f"Gesture: {gesture}", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                        self.control_mouse(gesture, pos, roll, image)

            # Visual Debug
            self.draw_debug_window()

            cv2.imshow('Eye & Gesture Control', image)
            if cv2.waitKey(5) & 0xFF == 27: break
                
        self.cap.release()
        cv2.destroyAllWindows()

# --- Fonction principale ---
if __name__ == "__main__":
    controller = EyeGestureController()
    controller.run()
