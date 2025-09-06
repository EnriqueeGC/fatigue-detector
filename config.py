
"""
Archivo de configuración para centralizar todos los parámetros ajustables
del detector de parpadeos.
"""

# --- Parámetros de Detección de Parpadeo ---
# Umbral de EAR por debajo del cual se considera que el ojo está cerrado.
EAR_THRESHOLD = 0.21
# Número de fotogramas consecutivos con EAR bajo para registrar un parpadeo.
CONSECUTIVE_FRAMES_THRESHOLD = 2

# --- Configuración del Modelo MediaPipe ---
# Número máximo de rostros a detectar.
MAX_FACES = 1
# Confianza mínima para la detección inicial de rostros.
MIN_DETECTION_CONFIDENCE = 0.5
# Confianza mínima para el seguimiento de landmarks.
MIN_TRACKING_CONFIDENCE = 0.5

# --- Índices de Landmarks para los Ojos ---
# Modelo de MediaPipe Face Mesh: https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png
# [P1, P2, P3, P4, P5, P6]
LEFT_EYE_INDEXES = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_INDEXES = [33, 160, 158, 133, 153, 144]

# --- Configuración de Visualización (OpenCV) ---
FONT = "FONT_HERSHEY_SIMPLEX"
FONT_SCALE_INFO = 0.7
FONT_THICKNESS_INFO = 2
COLOR_EAR = (0, 0, 255)  # Rojo en BGR
COLOR_COUNTER = (255, 0, 0) # Azul en BGR
TEXT_POSITION_COUNTER = (30, 50)
TEXT_POSITION_EAR = (30, 90)

# Radio y color para dibujar los landmarks de los ojos
LANDMARK_DRAW_RADIUS = 2
LANDMARK_DRAW_COLOR = (0, 255, 255) # Amarillo en BGR

# --- Parámetros de Detección de Parpadeo ---
# Umbral de EAR por debajo del cual se considera que el ojo está cerrado.
# EAR_THRESHOLD = 0.21
# Número de fotogramas consecutivos con EAR bajo para registrar un parpadeo normal.
CONSECUTIVE_FRAMES_THRESHOLD = 2
# Número de fotogramas consecutivos con EAR bajo para registrar un parpadeo LARGO (somnolencia).
# Ajusta este valor según tus pruebas. Un segundo a 30 FPS serían 30 fotogramas.
LONG_BLINK_DURATION_FRAMES = 30 # Ejemplo: para 1 segundo a 30 FPS

LONG_BLINK_FRAMES_THRESHOLD = 30 # Aproximadamente 1 segundo a 30 FPS


# --- Parámetros de Detección de Parpadeo BASADOS EN TIEMPO ---
# Umbral de EAR por debajo del cual se considera que el ojo está cerrado.
EAR_THRESHOLD = 0.22  # Puedes ajustar este valor

# Duración MÍNIMA para un parpadeo normal (en segundos). Menos que esto se ignora.
MIN_BLINK_DURATION_SECONDS = 0.1

# Duración MÁXIMA para un parpadeo normal (en segundos).
MAX_NORMAL_BLINK_DURATION_SECONDS = 0.5

# Duración MÍNIMA para que un parpadeo se considere LARGO (fatiga).
LONG_BLINK_DURATION_SECONDS = 1.0 # Si el ojo está cerrado por 1 segundo o más

# --- Parámetros de Precisión y Calidad de Datos ---
# Umbral mínimo de EAR para considerar que la detección de un ojo es válida.
# Ayuda a descartar fotogramas donde la cara está muy de perfil.
VALID_EAR_THRESHOLD = 0.05

# Texto a mostrar cuando la posición del rostro no es válida para la detección.
INVALID_POSE_TEXT = "Mire de Frente"