import numpy as np

def calculate_ear(eye_landmarks, frame_shape):
    """
    Calcula el Eye Aspect Ratio (EAR) para un solo ojo.

    Args:
        eye_landmarks: Lista de landmarks del ojo proporcionados por MediaPipe.
        frame_shape: Tupla (altura, anchura) del frame de la cámara.

    Returns:
        float: El valor de EAR calculado.
    """
    # Extraer las coordenadas (x, y) de los landmarks del ojo, convirtiéndolas
    # de relativas (0-1) a absolutas (píxeles).
    p2_y = int(eye_landmarks[1].y * frame_shape[0])
    p6_y = int(eye_landmarks[5].y * frame_shape[0])
    p3_y = int(eye_landmarks[2].y * frame_shape[0])
    p5_y = int(eye_landmarks[4].y * frame_shape[0])

    p1_x = int(eye_landmarks[0].x * frame_shape[1])
    p4_x = int(eye_landmarks[3].x * frame_shape[1])

    # --- CORRECCIÓN AQUÍ ---
    # Convertimos las tuplas a arrays de NumPy para poder realizar la resta vectorial.
    vertical_dist1 = np.linalg.norm(np.array([0, p2_y]) - np.array([0, p6_y]))
    vertical_dist2 = np.linalg.norm(np.array([0, p3_y]) - np.array([0, p5_y]))
    horizontal_dist = np.linalg.norm(np.array([p1_x, 0]) - np.array([p4_x, 0]))

    # Si la distancia horizontal es cero, evitamos la división por cero
    if horizontal_dist == 0:
        return 0.0

    # Fórmula EAR
    ear = (vertical_dist1 + vertical_dist2) / (2.0 * horizontal_dist)
    return ear