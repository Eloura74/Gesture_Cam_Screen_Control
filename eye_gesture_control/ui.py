import cv2
import numpy as np
import time
import math
from config import *
from ui_elements import Button

class UIManager:
    def __init__(self):
        # Initialisation des boutons (positions dynamiques calculées au draw si besoin, 
        # mais ici on fixe pour simplifier ou on update dans draw)
        # On les place en bas de l'écran par défaut
        self.buttons = []
        # On initialisera les boutons au premier appel de draw car on a besoin de w/h
        self.buttons_initialized = False

    def init_buttons(self, w, h):
        # Layout: Centré en bas
        total_w = 3 * BTN_WIDTH + 2 * BTN_MARGIN
        start_x = (w - total_w) // 2
        y_pos = h - BTN_HEIGHT - 20

        self.buttons = [
            Button("CALIBRATE", start_x, y_pos, BTN_WIDTH, BTN_HEIGHT, "CALIBRATE"),
            Button("PAUSE APP", start_x + BTN_WIDTH + BTN_MARGIN, y_pos, BTN_WIDTH, BTN_HEIGHT, "PAUSE"),
            Button("EXIT", start_x + 2 * (BTN_WIDTH + BTN_MARGIN), y_pos, BTN_WIDTH, BTN_HEIGHT, "EXIT", color_hover=C_DANGER)
        ]
        self.buttons_initialized = True

    def handle_click(self, x, y):
        for btn in self.buttons:
            if btn.is_clicked(x, y):
                return btn.action_code
        return None

    def handle_mouse_move(self, x, y):
        for btn in self.buttons:
            btn.check_hover(x, y)

    def draw_glass_panel(self, img, pt1, pt2, color=C_GLASS, alpha=0.8):
        """Dessine un panneau style 'verre' semi-transparent"""
        overlay = img.copy()
        cv2.rectangle(overlay, pt1, pt2, color, -1)
        
        # Bordure fine
        cv2.rectangle(overlay, pt1, pt2, C_GLASS_BORDER, 1)
        
        # Coins Tech
        x1, y1 = pt1
        x2, y2 = pt2
        l = 8
        c_corn = C_ACCENT_CYAN
        cv2.line(img, (x1, y1), (x1+l, y1), c_corn, 1)
        cv2.line(img, (x1, y1), (x1, y1+l), c_corn, 1)
        cv2.line(img, (x2, y2), (x2-l, y2), c_corn, 1)
        cv2.line(img, (x2, y2), (x2, y2-l), c_corn, 1)
        
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    def draw_hud_header(self, image, fps):
        h, w, _ = image.shape
        if not self.buttons_initialized:
            self.init_buttons(w, h)

        # Top Bar Background
        self.draw_glass_panel(image, (0, 0), (w, 45), C_BACKGROUND_DARK, 0.9)
        
        # Titre & FPS
        cv2.putText(image, "EYE.OS // PRO", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, C_ACCENT_CYAN, 2)
        
        # Indicateur Pulse
        pulse = abs(math.sin(time.time() * 2))
        col_pulse = (0, int(255*pulse), 0)
        cv2.circle(image, (w-120, 22), 4, col_pulse, -1)
        cv2.putText(image, f"{int(fps)} FPS", (w-100, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, C_TEXT_DIM, 1)

        # Dessin des boutons
        for btn in self.buttons:
            btn.draw(image)

    def draw_hud_sidebar(self, image, yaw, pitch, last_log_message, start_time):
        h, w, _ = image.shape
        # Panneau Gauche: Logs
        self.draw_glass_panel(image, (10, h-150), (350, h-80), C_GLASS, 0.6)
        cv2.putText(image, f">> {last_log_message}", (20, h-120), cv2.FONT_HERSHEY_SIMPLEX, 0.45, C_TEXT_MAIN, 1)
        
        # Timer
        uptime = int(time.time() - start_time)
        m, s = divmod(uptime, 60)
        cv2.putText(image, f"SESSION: {m:02}:{s:02}", (20, h-100), cv2.FONT_HERSHEY_SIMPLEX, 0.4, C_ACCENT_CYAN, 1)

        # Panneau Droite: Radar
        cx, cy = w-95, h-95
        self.draw_glass_panel(image, (w-180, h-180), (w-10, h-80), C_GLASS, 0.6)
        
        # Radar Crosshair
        cv2.line(image, (cx-60, cy), (cx+60, cy), C_GLASS_BORDER, 1)
        cv2.line(image, (cx, cy-60), (cx, cy+60), C_GLASS_BORDER, 1)
        cv2.circle(image, (cx, cy), 40, C_GLASS_BORDER, 1)
        
        # Radar Gaze Dot (Cible)
        gx = int(cx + (yaw * 1.5))
        gy = int(cy - (pitch * 1.5))
        cv2.line(image, (cx, cy), (gx, gy), C_ACCENT_ORANGE, 1)
        cv2.circle(image, (gx, gy), 3, C_ACCENT_ORANGE, -1)

    def draw_face_hud(self, image, landmarks, is_afk, calibration_mode):
        """Dessine un cadre de verrouillage autour du visage"""
        h, w, _ = image.shape
        
        xs = [lm.x for lm in landmarks]
        ys = [lm.y for lm in landmarks]
        x1, y1 = int(min(xs)*w), int(min(ys)*h)
        x2, y2 = int(max(xs)*w), int(max(ys)*h)
        
        pad = 25
        x1, y1 = max(0, x1-pad), max(0, y1-pad)
        x2, y2 = min(w, x2+pad), min(h, y2+pad)
        
        color = C_DANGER if is_afk else C_ACCENT_CYAN
        if calibration_mode: color = C_ACCENT_ORANGE

        # Coins du cadre (Bracket style)
        sl = 25 
        thickness = 2
        # Haut Gauche
        cv2.line(image, (x1, y1), (x1+sl, y1), color, thickness)
        cv2.line(image, (x1, y1), (x1, y1+sl), color, thickness)
        # Haut Droite
        cv2.line(image, (x2, y1), (x2-sl, y1), color, thickness)
        cv2.line(image, (x2, y1), (x2, y1+sl), color, thickness)
        # Bas Gauche
        cv2.line(image, (x1, y2), (x1+sl, y2), color, thickness)
        cv2.line(image, (x1, y2), (x1, y2-sl), color, thickness)
        # Bas Droite
        cv2.line(image, (x2, y2), (x2-sl, y2), color, thickness)
        cv2.line(image, (x2, y2), (x2, y2-sl), color, thickness)
        
        # Label au dessus
        label = "TARGET LOCKED"
        if is_afk: label = "TARGET LOST"
        if calibration_mode: label = "CALIBRATING..."
        
        cv2.putText(image, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    def draw_status_overlay(self, image, current_screen, calibration_mode, is_afk, last_swipe_time, last_swipe_direction):
        h, w, _ = image.shape
        cx, cy = w//2, h//2
        
        # Affichage Ecran Actif en Haut
        screen_label = current_screen.replace("ECRAN_", "").replace("_", " ")
        self.draw_glass_panel(image, (cx-120, 50), (cx+120, 85), C_GLASS, 0.9)
        cv2.putText(image, screen_label, (cx-100, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, C_ACCENT_CYAN, 1)

        # Mode Calibration
        if calibration_mode:
            cv2.rectangle(image, (0,0), (w,h), C_ACCENT_ORANGE, 10)
            self.draw_glass_panel(image, (cx-200, cy-50), (cx+200, cy+50), C_BACKGROUND_DARK, 0.9)
            cv2.putText(image, "CALIBRATION MODE", (cx-140, cy-10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, C_ACCENT_ORANGE, 2)
            cv2.putText(image, "LOOK AT TARGET & PRESS SPACE", (cx-160, cy+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, C_TEXT_MAIN, 1)

        # Mode AFK
        if is_afk:
            overlay = image.copy()
            cv2.rectangle(overlay, (0,0), (w,h), (0,0,20), -1) 
            cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)
            cv2.putText(image, "SYSTEM LOCKED / AFK", (cx-180, cy), cv2.FONT_HERSHEY_SIMPLEX, 1.0, C_DANGER, 2)
            
        # Feedback Swipe
        if time.time() - last_swipe_time < 1.0 and last_swipe_direction:
            text = f"SWIPE {last_swipe_direction} >>"
            self.draw_glass_panel(image, (cx-120, cy+100), (cx+120, cy+140), C_GLASS, 0.9)
            cv2.putText(image, text, (cx-100, cy+128), cv2.FONT_HERSHEY_SIMPLEX, 0.7, C_SUCCESS, 2)

    def draw_gesture_feedback(self, image, gesture, pos, landmarks, roll_angle, scroll_speed=0):
        if gesture != "NONE":
            h, w, _ = image.shape
            if pos:
                cx, cy = int(pos.x * w), int(pos.y * h)
            else:
                cx, cy = int(landmarks[0].x * w), int(landmarks[0].y * h)

            # Petit label flottant
            cv2.putText(image, gesture, (cx + 20, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, C_ACCENT_CYAN, 1)

            CONNECTIONS = [
                (0,1), (1,2), (2,3), (3,4), (0,5), (5,6), (6,7), (7,8),
                (5,9), (9,10), (10,11), (11,12), (9,13), (13,14), (14,15), (15,16),
                (13,17), (17,18), (18,19), (19,20), (0,17)
            ]
            for s, e in CONNECTIONS:
                x1, y1 = int(landmarks[s].x * w), int(landmarks[s].y * h)
                x2, y2 = int(landmarks[e].x * w), int(landmarks[e].y * h)
                cv2.line(image, (x1, y1), (x2, y2), C_GLASS_BORDER, 1)
        
        if gesture == "THREE_FINGERS" and pos:
            h, w, _ = image.shape
            cx, cy = int(pos.x * w), int(pos.y * h)
            color_joy = C_ACCENT_CYAN
            DEADZONE = 15
            if abs(roll_angle) > DEADZONE: color_joy = C_SUCCESS
            
            cv2.circle(image, (cx, cy), 45, color_joy, 2)
            
            rad = math.radians(roll_angle - 90)
            ex = int(cx + 45 * math.cos(rad))
            ey = int(cy + 45 * math.sin(rad))
            cv2.line(image, (cx, cy), (ex, ey), color_joy, 2)
            
            cv2.putText(image, f"SCROLL {scroll_speed}", (cx-35, cy-55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_joy, 1)
