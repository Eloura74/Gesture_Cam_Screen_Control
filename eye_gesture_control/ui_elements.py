from config import *
from ui_utils import draw_rounded_rect, draw_text_centered, FONT_MD

class Button:
    def __init__(self, text, x, y, w, h, action_code, color_base=C_BG_LIGHT, color_hover=C_ACCENT_BLUE):
        self.text = text
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.action_code = action_code
        self.color_base = color_base
        self.color_hover = color_hover
        self.is_hovered = False

    def draw(self, draw):
        # Couleur dynamique avec effet de "lueur" si survolé
        main_col = self.color_hover if self.is_hovered else self.color_base
        border_col = self.color_hover if self.is_hovered else (100, 100, 100, 150)
        
        # 1. Fond semi-transparent (Glassmorphism)
        draw_rounded_rect(draw, (self.x, self.y, self.x + self.w, self.y + self.h), (20, 22, 26, 160), radius=5)
        
        # 2. Bordure fine
        draw.rectangle([self.x, self.y, self.x + self.w, self.y + self.h], outline=border_col, width=1)

        # 3. Accents de coins (Style Militaire/Sci-fi)
        l = 10  # Longueur de l'accent
        # Top-Left
        draw.line((self.x, self.y, self.x + l, self.y), fill=main_col, width=2)
        draw.line((self.x, self.y, self.x, self.y + l), fill=main_col, width=2)
        # Bottom-Right
        draw.line((self.x + self.w, self.y + self.h, self.x + self.w - l, self.y + self.h), fill=main_col, width=2)
        draw.line((self.x + self.w, self.y + self.h, self.x + self.w, self.y + self.h - l), fill=main_col, width=2)

        # 4. Texte avec légère ombre pour la lisibilité
        cx = self.x + self.w / 2
        cy = self.y + self.h / 2
        draw_text_centered(draw, (cx, cy), self.text, FONT_MD, C_TEXT_WHITE if not self.is_hovered else main_col)

    def check_hover(self, mx, my):
        self.is_hovered = (self.x <= mx <= self.x + self.w) and (self.y <= my <= self.y + self.h)
        return self.is_hovered

    def is_clicked(self, mx, my):
        return (self.x <= mx <= self.x + self.w) and (self.y <= my <= self.y + self.h)
