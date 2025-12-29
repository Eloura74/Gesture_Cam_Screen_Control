import cv2
from config import *

class Button:
    def __init__(self, text, x, y, w, h, action_code, color_base=C_GLASS, color_hover=C_GLASS_BORDER):
        self.text = text
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.action_code = action_code
        self.color_base = color_base
        self.color_hover = color_hover
        self.is_hovered = False

    def draw(self, image):
        # Couleur selon état
        bg_color = self.color_hover if self.is_hovered else self.color_base
        border_color = C_ACCENT_CYAN if self.is_hovered else C_GLASS_BORDER
        text_color = C_ACCENT_CYAN if self.is_hovered else C_TEXT_MAIN

        # Fond
        cv2.rectangle(image, (self.x, self.y), (self.x + self.w, self.y + self.h), bg_color, -1)
        
        # Bordure "Tech" (Coins coupés ou lignes)
        cv2.rectangle(image, (self.x, self.y), (self.x + self.w, self.y + self.h), border_color, 1)
        
        # Petite déco "Tech" sur le coté gauche
        cv2.line(image, (self.x, self.y + 5), (self.x, self.y + self.h - 5), C_ACCENT_CYAN, 2)

        # Texte Centré
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.5
        thickness = 1
        (text_w, text_h), _ = cv2.getTextSize(self.text, font, scale, thickness)
        
        tx = self.x + (self.w - text_w) // 2
        ty = self.y + (self.h + text_h) // 2
        
        cv2.putText(image, self.text, (tx, ty), font, scale, text_color, thickness, cv2.LINE_AA)

    def check_hover(self, mx, my):
        self.is_hovered = (self.x <= mx <= self.x + self.w) and (self.y <= my <= self.y + self.h)
        return self.is_hovered

    def is_clicked(self, mx, my):
        return (self.x <= mx <= self.x + self.w) and (self.y <= my <= self.y + self.h)
