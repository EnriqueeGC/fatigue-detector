
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

# Umbral de parpadeos largos para activar la alerta fuerte
MAX_LONG_BLINKS_FOR_STRONG_ALERT = 3

# --- Detección de Bostezos ---
# Índices de los landmarks de la boca en el modelo de MediaPipe
# Contorno exterior completo de la boca (más robusto)
MOUTH_INDEXES = [
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
    291, 375, 321, 405, 314, 17, 84, 181, 91, 146
]
# También podemos incluir algunos puntos internos para calcular el MAR más preciso
# Los puntos 13 (arriba) y 14 (abajo) pueden ser útiles para la altura
# Los puntos 78 (izquierda) y 308 (derecha) para el ancho
MOUTH_INDEXES_FOR_MAR_CALC = [13, 14, 78, 308] # Estos se usarán específicamente para calculate_mar

YAWN_THRESHOLD = 0.5  # Umbral para considerar la boca abierta (ajustar según pruebas)
MIN_YAWN_DURATION_SECONDS = 1.0  # Duración mínima de un bostezo para ser contado

# Posiciones de texto para la visualización
TEXT_POSITION_YAWN_COUNTER = (20, 160) # Ajustado para no sobreponerse con la alerta
TEXT_POSITION_MAR = (20, 190) # Ajustado

# Colores y tamaños para la visualización
COLOR_YAWN_COUNTER = (0, 0, 255)  # Rojo
COLOR_MAR = (255, 128, 0) # Naranja

# --- Configuración para la Alerta de Bostezos ---
YAWN_ALERT_TIME_WINDOW = 60  # Duración de la ventana de tiempo en segundos (ej. 60s = 1 minuto)
YAWN_ALERT_THRESHOLD = 3      # Número de bostezos en la ventana de tiempo para activar la alerta
YAWN_ALERT_WINDOW_SIZE = YAWN_ALERT_THRESHOLD + 2 # Se recomienda que la cola sea un poco más grande que el umbral

# Ventana de tiempo para el suavizado del EAR (en número de fotogramas)
SMOOTHING_FRAMES = 8 

# --- Configuración para la calibración del umbral adaptativo ---
CALIBRATION_DURATION_SECONDS = 5.0 # Duración de la fase de calibración
CLOSED_EYE_RATIO = 0.70            # Porcentaje del EAR de ojos abiertos para definir el umbral de cierre
DROWSY_RATIO = 0.50                # Porcentaje del EAR de ojos abiertos para una alerta de somnolencia