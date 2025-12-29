import time
import math
import cv2
import numpy as np
import mediapipe as mp
import pyautogui
import pygetwindow as gw

from config import *
from utils import Stabilizer
from workers import ActionWorker
from camera import CameraManager
from calibration import CalibrationManager
from head_tracking import HeadTracker
from hand_tracking import HandTracker
from ui import UIManager
from actions import ScrollHandler, SwipeHandler, MediaHandler, MouseHandler

class EyeGestureController:
    def __init__(self):
        self.running = True
        self.start_time = time.time()
        self.frame_count = 0
        
        # --- Modules ---
        self.worker = ActionWorker()
        self.worker.start()
        
        self.camera = CameraManager()
        self.calibration = CalibrationManager(log_callback=self.log_action)
        self.head_tracker = HeadTracker()
        self.hand_tracker = HandTracker()
        self.ui = UIManager()
        
        # --- Action Handlers ---
        self.scroll_handler = ScrollHandler(self.worker)
        self.swipe_handler = SwipeHandler(self.worker, self.log_action)
        self.media_handler = MediaHandler(self.worker, self.log_action)
        self.mouse_handler = MouseHandler(self.worker)
        
        # --- État Système ---
        self.current_screen = "INCONNU"
        self.active_monitor = self.calibration.monitors[0] if self.calibration.monitors else None
        self.system_paused = False # Global pause state
        
        # Stabilisateurs Tête
        self.yaw_stabilizer = Stabilizer(HEAD_SMOOTHING)
        self.pitch_stabilizer = Stabilizer(HEAD_SMOOTHING)
        
        self.last_log_message = "SYSTEM INITIALIZED"
        self.last_log_time = time.time()
        
        self.last_face_detected_time = time.time()
        self.is_afk = False

        # --- Mouse Callback State ---
        self.mouse_x = 0
        self.mouse_y = 0
        self.click_event = None

        print(f"Système initialisé. {len(self.calibration.monitors)} moniteurs détectés.")

    def log_action(self, message):
        print(f"[ACTION] {message}")
        self.last_log_message = message
        self.last_log_time = time.time()

    def on_mouse(self, event, x, y, flags, param):
        self.mouse_x, self.mouse_y = x, y
        if event == cv2.EVENT_LBUTTONDOWN:
            self.click_event = (x, y)

    def handle_ui_interactions(self):
        # Hover update
        self.ui.handle_mouse_move(self.mouse_x, self.mouse_y)
        
        # Click handling
        if self.click_event:
            cx, cy = self.click_event
            action = self.ui.handle_click(cx, cy)
            if action:
                self.log_action(f"BUTTON CLICK: {action}")
                if action == "CALIBRATE":
                    self.calibration.calibration_mode = not self.calibration.calibration_mode
                    self.calibration.calibration_step = 0
                elif action == "PAUSE":
                    self.system_paused = not self.system_paused
                    # Update button text
                    for btn in self.ui.buttons:
                        if btn.action_code == "PAUSE":
                            btn.text = "RESUME" if self.system_paused else "PAUSE APP"
                            btn.color_base = C_ACCENT_ORANGE if self.system_paused else C_GLASS
                    
                    self.log_action(f"SYSTEM {'PAUSED' if self.system_paused else 'RESUMED'}")
                    
                elif action == "EXIT":
                    self.running = False
            
            self.click_event = None # Reset

    def determine_screen(self, yaw, pitch):
        best_screen = self.current_screen
        min_dist = float('inf')
        hysteresis_bonus = 5.0 

        for screen_name, data in self.calibration.screen_centers.items():
            c_yaw = data["yaw"]
            c_pitch = data["pitch"]
            dist = math.sqrt((yaw - c_yaw)**2 + (pitch - c_pitch)**2)
            if screen_name == self.current_screen:
                dist -= hysteresis_bonus
            if dist < min_dist:
                min_dist = dist
                best_screen = screen_name
        return best_screen

    def update_active_monitor(self, screen_name):
        self.active_monitor = self.calibration.get_active_monitor(screen_name)

    def process_actions(self, gesture, position, roll_angle, landmarks, image=None):
        # Actions Ecrans
        if self.current_screen == "ECRAN_3_HAUT":
            self.media_handler.handle_fist(gesture, self.active_monitor)
            return
        
        # Si on change d'écran, on reset le lock pour ne pas rester coincé
        self.media_handler.reset_lock()

        if self.current_screen == "ECRAN_1_GAUCHE":
            is_horizontal = abs(roll_angle) > 30
            if gesture == "TWO_FINGERS" and position and is_horizontal:
                swipe = self.hand_tracker.detect_swipe(position)
                if swipe:
                    self.swipe_handler.handle(swipe)
                return

        speed = 0
        if gesture == "THREE_FINGERS" and position:
            speed = self.scroll_handler.handle(roll_angle)
        
        # Feedback visuel pour le scroll
        if image is not None and gesture == "THREE_FINGERS":
             self.ui.draw_gesture_feedback(image, gesture, position, landmarks, roll_angle, speed)
             return
        
        if gesture == "POINT" and position:
            self.mouse_handler.handle(position, self.active_monitor)

    def shutdown(self):
        """Libère proprement toutes les ressources."""
        if hasattr(self, "worker") and self.worker:
            self.worker.stop()
        
        if hasattr(self, "camera") and self.camera:
            self.camera.release()

        cv2.destroyAllWindows()

        if hasattr(self, "head_tracker") and self.head_tracker:
            self.head_tracker.close()

        if hasattr(self, "hand_tracker") and self.hand_tracker:
            self.hand_tracker.close()

    def run(self):
        prev_time = time.time()
        
        # Setup Window & Mouse Callback
        cv2.namedWindow('EyeOS v2.4 [CYBERPUNK]', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('EyeOS v2.4 [CYBERPUNK]', self.on_mouse)

        try:
            while self.running:
                success, image = self.camera.read()
                if not success:
                    break

                self.frame_count += 1

                # FPS Calculation
                curr_time = time.time()
                fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 30
                prev_time = curr_time

                # Si le système est en pause, on affiche juste l'image et l'UI, sans processing IA
                if self.system_paused:
                    # Overlay Pause
                    overlay = image.copy()
                    cv2.rectangle(overlay, (0,0), (image.shape[1], image.shape[0]), (0,0,0), -1)
                    cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
                    
                    # UI de base (Header + Boutons pour pouvoir Resume)
                    self.ui.draw_hud_header(image, fps)
                    
                    # Gros texte PAUSE
                    h, w, _ = image.shape
                    cv2.putText(image, "SYSTEM PAUSED", (w//2 - 200, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1.5, C_ACCENT_ORANGE, 3)
                    cv2.putText(image, "Click RESUME to continue", (w//2 - 180, h//2 + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, C_TEXT_MAIN, 1)

                    # Interaction UI (pour le bouton Resume)
                    self.handle_ui_interactions()
                    
                    cv2.imshow('EyeOS v2.4 [CYBERPUNK]', image)
                    if cv2.waitKey(1) & 0xFF == 27: break
                    continue

                # --- NORMAL PROCESSING ---
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                )

                # --- Head Tracking ---
                try:
                    face_result = self.head_tracker.detect(mp_image)
                except KeyboardInterrupt:
                    raise
                
                yaw_stable, pitch_stable = 0, 0

                if face_result and face_result.face_landmarks:
                    self.last_face_detected_time = time.time()
                    if self.is_afk:
                        self.log_action("USER IDENTIFIED")
                        self.media_handler.trigger_play_pause(self.active_monitor)
                        self.is_afk = False

                    landmarks = face_result.face_landmarks[0]
                    self.ui.draw_face_hud(image, landmarks, self.is_afk, self.calibration.calibration_mode)

                    raw_yaw, raw_pitch = self.head_tracker.get_head_pose(image.shape, landmarks)
                    yaw_stable = self.yaw_stabilizer.update(raw_yaw)
                    pitch_stable = self.pitch_stabilizer.update(raw_pitch)

                    if not self.calibration.calibration_mode:
                        new_screen = self.determine_screen(yaw_stable, pitch_stable)
                        if new_screen != self.current_screen:
                            self.current_screen = new_screen
                            self.update_active_monitor(new_screen)

                            # Reset stabilizers
                            self.mouse_handler.reset_stabilizers()

                            # Reset lock poing
                            self.media_handler.reset_lock()

                else:
                    if not self.is_afk and (time.time() - self.last_face_detected_time > AFK_TIMEOUT):
                        self.log_action("NO USER - LOCKING")
                        self.media_handler.trigger_play_pause(self.active_monitor)
                        self.is_afk = True

                # --- Hand Tracking ---
                try:
                    hand_result = self.hand_tracker.detect(mp_image)
                except KeyboardInterrupt:
                    raise

                if hand_result and hand_result.hand_landmarks and not self.calibration.calibration_mode:
                    landmarks = hand_result.hand_landmarks[0]
                    gesture, pos, roll = self.hand_tracker.detect_hand_gesture(landmarks)
                    
                    # Process Actions
                    self.process_actions(gesture, pos, roll, landmarks, image)
                    
                    # Draw Gesture Feedback (si pas déjà fait dans process_actions pour THREE_FINGERS)
                    if gesture != "THREE_FINGERS":
                        self.ui.draw_gesture_feedback(image, gesture, pos, landmarks, roll)

                # --- UI LAYERS ---
                self.ui.draw_hud_header(image, fps)
                self.ui.draw_hud_sidebar(image, yaw_stable, pitch_stable, self.last_log_message, self.start_time)
                self.ui.draw_status_overlay(image, self.current_screen, self.calibration.calibration_mode, self.is_afk, self.hand_tracker.last_swipe_display_time, self.hand_tracker.last_swipe_direction)

                # --- Handle UI Interactions (Clicks) ---
                self.handle_ui_interactions()

                cv2.imshow('EyeOS v2.4 [CYBERPUNK]', image)

                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    break
                elif key == ord('c'):
                    self.calibration.calibration_mode = not self.calibration.calibration_mode
                    self.calibration.calibration_step = 0
                    self.log_action("CALIBRATION MODE")
                elif key == 32 and self.calibration.calibration_mode:
                    self.calibration.calibrate_step(yaw_stable, pitch_stable)
                elif key == ord('m'):
                    is_paused = self.mouse_handler.toggle_pause()
                    self.log_action(f"MOUSE {'LOCKED' if is_paused else 'UNLOCKED'}")

        except KeyboardInterrupt:
            print("\n[EXIT] KeyboardInterrupt reçu, arrêt propre...")
        finally:
            self.shutdown()
