from scipy.spatial import distance as dist
import numpy as np
import config # Importamos config para acceder a los nuevos índices

def calculate_mar(all_face_landmarks, image_dims): # Cambiado el nombre del parámetro
    """
    Calcula el Mouth Aspect Ratio (MAR) basado en los landmarks de la boca.
    
    Args:
        all_face_landmarks (list): Una lista de *todos* los objetos de landmarks de la cara.
        image_dims (tuple): Dimensiones de la imagen (altura, ancho).

    Returns:
        float: El valor del MAR.
    """
    height, width = image_dims

    # Extraemos solo los puntos necesarios para el cálculo del MAR
    mouth_points_for_mar = [all_face_landmarks[i] for i in config.MOUTH_INDEXES_FOR_MAR_CALC]

    # Convertimos los landmarks de MediaPipe a un array de NumPy
    # Asegúrate de que los puntos 13, 14, 78, 308 estén presentes y en el orden correcto si dependes de ellos
    # Para mayor robustez, mapeamos directamente los índices del config a un array.
    p_13 = all_face_landmarks[13]
    p_14 = all_face_landmarks[14]
    p_78 = all_face_landmarks[78]
    p_308 = all_face_landmarks[308]

    # Convertir a coordenadas pixel
    p_13_coords = np.array([int(p_13.x * width), int(p_13.y * height)])
    p_14_coords = np.array([int(p_14.x * width), int(p_14.y * height)])
    p_78_coords = np.array([int(p_78.x * width), int(p_78.y * height)])
    p_308_coords = np.array([int(p_308.x * width), int(p_308.y * height)])
    
    # Distancia vertical (altura de la boca)
    # Puntos 13 (labio superior central) y 14 (labio inferior central)
    vertical_dist = dist.euclidean(p_13_coords, p_14_coords)
    
    # Distancia horizontal (ancho de la boca)
    # Puntos 78 (comisura izquierda) y 308 (comisura derecha)
    horizontal_dist = dist.euclidean(p_78_coords, p_308_coords)

    # El MAR es la relación entre la altura y el ancho.
    # Se agrega una pequeña constante al denominador para evitar división por cero.
    mar = vertical_dist / (horizontal_dist + 1e-6) # Añadir épsilon para evitar ZeroDivisionError

    return mar