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

# --- PALETTE "CYBER-GLASS" (BGR) ---
C_CYAN_DIM = (100, 100, 0)
C_CYAN_BRIGHT = (255, 255, 100)
C_MAGENTA_DEEP = (100, 0, 100)
C_MAGENTA_NEON = (255, 0, 255)
C_BLUE_TECH = (255, 150, 0)
C_GREEN_DATA = (50, 255, 100)
C_ALERT_RED = (0, 0, 255)
C_WHITE = (240, 240, 255)
C_GLASS_DARK = (10, 15, 20)
