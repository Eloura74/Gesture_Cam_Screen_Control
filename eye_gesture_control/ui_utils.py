from PIL import Image, ImageDraw, ImageFont
from config import *
import os

def load_font(path, size):
    try:
        return ImageFont.truetype(path, size)
    except IOError:
        try:
            return ImageFont.truetype(FONT_FALLBACK, size)
        except IOError:
            return ImageFont.load_default()

# Pré-chargement des polices courantes
FONT_SM = load_font(FONT_MAIN, 12)
FONT_MD = load_font(FONT_MAIN, 16)
FONT_LG = load_font(FONT_BOLD, 24)
FONT_XL = load_font(FONT_BOLD, 32)
FONT_ICON = load_font(FONT_BOLD, 20)

def draw_rounded_rect(draw, xy, color, radius=10, border_color=None, border_width=0):
    """Dessine un rectangle arrondi"""
    x1, y1, x2, y2 = xy
    draw.rounded_rectangle((x1, y1, x2, y2), radius=radius, fill=color, outline=border_color, width=border_width)

def draw_text_centered(draw, xy, text, font, color):
    """Dessine du texte centré en (x, y)"""
    cx, cy = xy
    # getbbox retourne (left, top, right, bottom)
    bbox = font.getbbox(text)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    draw.text((cx - w / 2, cy - h / 2 - bbox[1]), text, font=font, fill=color)

def draw_glass_panel(draw, xy, color, radius=15):
    """Dessine un panneau semi-transparent (simulation)"""
    # Note: PIL draw direct ne gère pas l'alpha sur l'image de base si on dessine avec une couleur RGBA sur RGB.
    # Il faut idéalement dessiner sur un layer transparent puis alpha_composite.
    # Pour simplifier ici, on suppose que 'draw' est sur un layer RGBA.
    draw.rounded_rectangle(xy, radius=radius, fill=color)
