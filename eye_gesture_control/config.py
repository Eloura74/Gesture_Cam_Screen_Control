# --- CONFIGURATION GLOBALE ---
CONFIG_FILE = "screen_calibration.json"
# Note sur Alpha : 1.0 = aucune stabilisation (brut), 0.1 = très fort lissage (lent)
MOUSE_SMOOTHING = 0.4  
HEAD_SMOOTHING = 0.8
SCROLL_SPEED = 30
SWIPE_THRESHOLD = 0.04 
SWIPE_TIME_WINDOW = 0.6
SWIPE_COOLDOWN = 1.0
VOLUME_DEBOUNCE = 0.2
AFK_TIMEOUT = 5.0
PAUSE_COOLDOWN = 1.5

# --- PALETTE "MODERN CLEAN TECH" (RGB pour PIL) ---
# Note: PIL utilise RGB, OpenCV utilise BGR. 
# Ici on définit en RGB pour PIL.
C_BG_DARK = (20, 22, 26)       # Gris très sombre, presque noir (Fond panneaux)
C_BG_LIGHT = (40, 44, 52)      # Gris un peu plus clair (Fond boutons)
C_ACCENT_BLUE = (0, 120, 215)  # "Electric Blue" (Windows 10/11 style)
C_ACCENT_CYAN = (0, 255, 255)  # Cyan vif pour les highlights
C_ACCENT_WARN = (255, 165, 0)  # Orange
C_ACCENT_ERR = (220, 50, 50)   # Rouge doux
C_TEXT_WHITE = (255, 255, 255) # Blanc pur
C_TEXT_GREY = (180, 180, 180)  # Gris clair

# --- FONTS ---
# Chemins Windows standards
FONT_MAIN = "C:/Windows/Fonts/segoeui.ttf"
FONT_BOLD = "C:/Windows/Fonts/segoeuib.ttf"
# Fallback si pas trouvé
FONT_FALLBACK = "arial.ttf"

# --- BUTTONS CONFIG ---
BTN_HEIGHT = 45
BTN_WIDTH = 140
BTN_MARGIN = 15
BTN_RADIUS = 10 # Rayon des coins arrondis
