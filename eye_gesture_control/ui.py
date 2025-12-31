import cv2
import numpy as np
import time
import math
import random
from PIL import Image, ImageDraw
from config import *
from ui_elements import Button
from ui_utils import *

class UIManager:
    def __init__(self):
        self.buttons = []
        self.buttons_initialized = False
        self.scan_line_y = 0 
        self.noise_overlay = None

    def init_buttons(self, w, h):
        total_w = 4 * BTN_WIDTH + 3 * BTN_MARGIN
        start_x = (w - total_w) // 2
        y_pos = h - BTN_HEIGHT - 30 

        self.buttons = [
            Button("CALIBRATE", start_x, y_pos, BTN_WIDTH, BTN_HEIGHT, "CALIBRATE"),
            Button("RECENTER", start_x + BTN_WIDTH + BTN_MARGIN, y_pos, BTN_WIDTH, BTN_HEIGHT, "RECENTER", color_hover=C_ACCENT_CYAN),
            Button("PAUSE APP", start_x + 2 * (BTN_WIDTH + BTN_MARGIN), y_pos, BTN_WIDTH, BTN_HEIGHT, "PAUSE"),
            Button("EXIT", start_x + 3 * (BTN_WIDTH + BTN_MARGIN), y_pos, BTN_WIDTH, BTN_HEIGHT, "EXIT", color_hover=C_ACCENT_ERR)
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

    def draw_tech_border(self, draw, x1, y1, x2, y2, color, label=None):
        # Fond plus transparent pour éviter de bloquer la vue
        draw.rectangle([x1, y1, x2, y2], fill=(10, 12, 15, 120))
        draw.rectangle([x1, y1, x2, y2], outline=(color[0], color[1], color[2], 80), width=1)
        cl = 15
        draw.line((x1, y1, x1+cl, y1), fill=color, width=2) 
        draw.line((x1, y1, x1, y1+cl), fill=color, width=2)
        draw.line((x2, y2, x2-cl, y2), fill=color, width=2) 
        draw.line((x2, y2, x2, y2-cl), fill=color, width=2)
        if label:
            draw.text((x1 + 5, y1 - 18), label, font=FONT_SM, fill=color)

    def draw_setup_visualizer(self, draw, x, y, w, h, current_screen, yaw, pitch):
        """Dessine la configuration physique des écrans (Layout Pro)"""
        
        # Layout:
        #       [TOP]
        # [LEFT][CTR ][RIGHT(V)]
        
        r_ctr = (0.33, 0.4, 0.33, 0.3)
        r_left = (0.0, 0.4, 0.33, 0.3)
        r_top = (0.33, 0.1, 0.33, 0.3)
        r_right = (0.66, 0.2, 0.2, 0.5)
        
        screens = {
            "ECRAN_1_GAUCHE": r_left,
            "ECRAN_2_CENTRE": r_ctr,
            "ECRAN_3_HAUT": r_top,
            "ECRAN_4_DROITE": r_right
        }
        
        # Fond du container
        draw_rounded_rect(draw, (x, y, x+w, y+h), (20, 22, 26, 150), radius=10)
        draw.text((x+10, y+5), "SYSTEM_LAYOUT", font=FONT_SM, fill=C_TEXT_GREY)

        for name, rect in screens.items():
            rx, ry, rw, rh = rect
            sx = x + int(rx * w) + 5
            sy = y + int(ry * h) + 20
            sw = int(rw * w) - 5
            sh = int(rh * h) - 5
            
            is_active = (name == current_screen)
            
            fill_col = (0, 120, 215, 120) if is_active else (40, 44, 52, 80)
            border_col = C_ACCENT_CYAN if is_active else C_TEXT_GREY
            width = 2 if is_active else 1
            
            draw.rectangle([sx, sy, sx+sw, sy+sh], fill=fill_col, outline=border_col, width=width)
            
            num = name.split("_")[1]
            draw_text_centered(draw, (sx + sw//2, sy + sh//2), num, FONT_MD, C_TEXT_WHITE if is_active else C_TEXT_GREY)
            
            if is_active:
                nx = np.clip((yaw + 30) / 60, 0, 1)
                ny = np.clip((-pitch + 20) / 40, 0, 1)
                gx = sx + int(nx * sw)
                gy = sy + int(ny * sh)
                draw.ellipse((gx-2, gy-2, gx+2, gy+2), fill=C_ACCENT_WARN)

    def generate_noise(self, w, h):
        # Génère une image de bruit statique une seule fois ou rarement pour perf
        noise = np.random.randint(0, 50, (h, w, 3), dtype=np.uint8)
        return Image.fromarray(noise).convert("RGBA")

    def draw_ui(self, cv2_image, fps, yaw, pitch, last_log, start_time, current_screen, calib_mode, is_afk, swipe_time, swipe_dir, landmarks=None, gesture=None, pos=None, roll=0, speed=0):
        # 1. Ajout de bruit/grain sur l'image source (OpenCV) pour l'effet "Caméra de surveillance"
        # On le fait léger pour ne pas tuer les perfs
        # noise = np.zeros(cv2_image.shape, dtype=np.uint8)
        # cv2.randn(noise, 0, 10)
        # cv2_image = cv2.add(cv2_image, noise)
        
        cv2_rgb = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(cv2_rgb).convert("RGBA")
        overlay = Image.new("RGBA", pil_img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        w, h = pil_img.size
        
        if not self.buttons_initialized:
            self.init_buttons(w, h)

        # --- SCANLINE ---
        self.scan_line_y = (self.scan_line_y + 3) % h
        # Ligne principale
        draw.line((0, self.scan_line_y, w, self.scan_line_y), fill=(0, 255, 255, 30), width=1)
        # Traînée
        draw.rectangle([0, self.scan_line_y-20, w, self.scan_line_y], fill=(0, 255, 255, 5))

        # --- HEADER (Plus discret) ---
        header_w = 400
        header_x = (w - header_w) // 2
        # Juste le texte et une ligne fine
        draw_text_centered(draw, (w//2, 25), "EYE.OS // NEURAL LINK", FONT_LG, C_TEXT_WHITE)
        draw.line((header_x, 45, header_x + header_w, 45), fill=C_ACCENT_BLUE, width=1)
        
        # FPS & Stats (Top Right)
        draw.text((w-100, 15), f"FPS: {int(fps)}", font=FONT_SM, fill=C_ACCENT_CYAN)
        draw.text((w-100, 30), f"LAT: {int(1000/fps if fps>0 else 0)}ms", font=FONT_SM, fill=C_TEXT_GREY)

        # --- LAYOUT REORGANISATION ---
        
        # 1. DATA LOG -> TOP LEFT (Pour libérer le bas)
        # Plus compact
        log_w = 250
        log_h = 100
        self.draw_tech_border(draw, 20, 60, 20 + log_w, 60 + log_h, C_ACCENT_BLUE, "SYSTEM_LOG")
        draw.text((30, 80), f"> STATUS: ACTIVE", font=FONT_SM, fill=C_ACCENT_CYAN)
        draw.text((30, 100), f"> {last_log[:28]}", font=FONT_SM, fill=C_TEXT_WHITE)
        draw.text((30, 125), f"DISPLAY: {current_screen.replace('ECRAN_', '')}", font=FONT_SM, fill=C_TEXT_GREY)

        # 2. SETUP VISUALIZER -> TOP RIGHT (Sous les FPS)
        viz_w = 200
        viz_h = 140
        vx = w - viz_w - 20
        vy = 60 # Alignement haut
        self.draw_setup_visualizer(draw, vx, vy, viz_w, viz_h, current_screen, yaw, pitch)

        # --- FACE BRACKETS (Subtils) ---
        if landmarks:
            xs = [lm.x for lm in landmarks]
            ys = [lm.y for lm in landmarks]
            x1, y1, x2, y2 = int(min(xs)*w), int(min(ys)*h), int(max(xs)*w), int(max(ys)*h)
            
            # Marge plus large pour "respirer"
            pad = 20
            x1, y1, x2, y2 = x1-pad, y1-pad, x2+pad, y2+pad
            
            col = C_ACCENT_ERR if is_afk else (C_ACCENT_WARN if calib_mode else C_ACCENT_CYAN)
            
            # Juste les coins (Corners)
            l = 20
            t = 2
            # TL
            draw.line((x1, y1, x1+l, y1), fill=col, width=t)
            draw.line((x1, y1, x1, y1+l), fill=col, width=t)
            # TR
            draw.line((x2, y1, x2-l, y1), fill=col, width=t)
            draw.line((x2, y1, x2, y1+l), fill=col, width=t)
            # BL
            draw.line((x1, y2, x1+l, y2), fill=col, width=t)
            draw.line((x1, y2, x1, y2-l), fill=col, width=t)
            # BR
            draw.line((x2, y2, x2-l, y2), fill=col, width=t)
            draw.line((x2, y2, x2, y2-l), fill=col, width=t)
            
            # Label discret au dessus
            draw.text((x1, y1 - 15), "SUBJECT_01" if not is_afk else "LOST", font=FONT_SM, fill=col)
            
            # Données à droite du cadre (plus petites)
            draw.text((x2 + 5, y1), f"Y:{yaw:.0f}", font=FONT_SM, fill=col)
            draw.text((x2 + 5, y1+12), f"P:{pitch:.0f}", font=FONT_SM, fill=col)

        # --- GESTURE ---
        if gesture and gesture != "NONE":
            if pos:
                gx, gy = int(pos.x * w), int(pos.y * h)
                draw.ellipse((gx-20, gy-20, gx+20, gy+20), outline=C_ACCENT_CYAN, width=2)
                draw.line((gx, gy-25, gx, gy+25), fill=C_ACCENT_CYAN, width=1)
                draw.line((gx-25, gy, gx+25, gy), fill=C_ACCENT_CYAN, width=1)
                draw.text((gx+25, gy-10), f"{gesture}", font=FONT_MD, fill=C_ACCENT_CYAN)

        # --- BOUTONS (Bas) ---
        for btn in self.buttons:
            btn.draw(draw)

        # --- VIGNETTE / GRAIN FINAL ---
        # Vignette simple (assombrissement coins)
        # On dessine un grand rectangle radial... trop lourd en PIL pur.
        # On fait juste un cadre sombre dégradé simulé par plusieurs rects transparents sur les bords
        border_w = 50
        # Top
        # draw.rectangle([0, 0, w, border_w], fill=(0,0,0,100))
        # Bottom
        # draw.rectangle([0, h-border_w, w, h], fill=(0,0,0,100))
        
        out = Image.alpha_composite(pil_img, overlay)
        return cv2.cvtColor(np.array(out.convert("RGB")), cv2.COLOR_RGB2BGR)

    def draw_hud_header(self, image, fps): pass 
    def draw_hud_sidebar(self, image, yaw, pitch, log, start): pass
    def draw_face_hud(self, image, lm, afk, calib): pass
    def draw_status_overlay(self, image, screen, calib, afk, st, sd): pass
    def draw_gesture_feedback(self, image, g, p, lm, r, s=0): pass
