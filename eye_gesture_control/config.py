# --- CONFIGURATION GLOBALE ---
CONFIG_FILE = "screen_calibration.json"
# Note sur Alpha : 1.0 = aucune stabilisation (brut), 0.1 = très fort lissage (lent)
MOUSE_SMOOTHING = 0.4  
HEAD_SMOOTHING = 0.8
SCROLL_SPEED = 30
SWIPE_THRESHOLD = 0.04 # Sensibilité augmentée (0.05 -> 0.04)
SWIPE_TIME_WINDOW = 0.6
SWIPE_COOLDOWN = 1.0
VOLUME_DEBOUNCE = 0.2
AFK_TIMEOUT = 5.0
PAUSE_COOLDOWN = 1.5  # Temps min entre deux actions

# --- PALETTE "PRO CYBER-GLASS" (BGR) ---
# Couleurs plus sobres, moins saturées, style "Interface Militaire / Sci-Fi"
C_BACKGROUND_DARK = (15, 15, 20)      # Fond très sombre presque noir
C_GLASS = (30, 35, 40)                # Verre fumé
C_GLASS_BORDER = (60, 70, 80)         # Bordure subtile
C_ACCENT_CYAN = (200, 200, 0)         # Cyan "Tech" (BGR: Blue-Green-Red -> 200,200,0 est Teal/Cyan)
C_ACCENT_ORANGE = (0, 140, 255)       # Orange "Alert" (BGR)
C_TEXT_MAIN = (220, 230, 240)         # Blanc cassé bleuté
C_TEXT_DIM = (120, 130, 140)          # Gris bleuté
C_SUCCESS = (50, 200, 50)             # Vert "Valid"
C_WARNING = (0, 165, 255)             # Orange
C_DANGER = (50, 50, 200)              # Rouge

# --- BUTTONS CONFIG ---
BTN_HEIGHT = 40
BTN_WIDTH = 120
BTN_MARGIN = 10
