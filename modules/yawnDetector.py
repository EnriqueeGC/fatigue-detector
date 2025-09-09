import cv2
import mediapipe as mp
import time

from utils.marDetector import calculate_mar
import config

class YawnDetector:
    def __init__(self, data_controller):
        """Inicializa el detector con el modelo de MediaPipe y el controlador de DB."""
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=config.MAX_FACES,
            refine_landmarks=True,
            min_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE
        )
        
        self.yawn_counter = 0
        self.yawn_start_time = None
        self.detection_reliable = True
        
        self.data_controller = data_controller

    def _update_yawn_counter(self, mar_value):
        """
        Actualiza el contador de bostezos basado en el valor MAR y el tiempo transcurrido.
        """
        if mar_value > config.YAWN_THRESHOLD:
            if self.yawn_start_time is None:
                self.yawn_start_time = time.time()
        else:
            if self.yawn_start_time is not None:
                duration = time.time() - self.yawn_start_time
                if duration >= config.MIN_YAWN_DURATION_SECONDS:
                    self.yawn_counter += 1
                    print(f"¡Bostezo Detectado! (Duración: {duration:.2f} segundos)")
                    self.data_controller.add_event_to_session(
                        event_type="bostezo",
                        description=f"Bostezo detectado. Duración: {duration:.2f} s."
                    )
                self.yawn_start_time = None

    def process_frame(self, frame):
        """
        Procesa un único fotograma para detectar bostezos.
        """
        height, width, _ = frame.shape
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(image_rgb)
        
        mar_value = -1.0
        self.detection_reliable = True
        
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0].landmark

            # Verificar si tenemos suficientes landmarks para los cálculos
            required_indices = set(config.MOUTH_INDEXES_FOR_MAR_CALC)
            if len(face_landmarks) > max(required_indices): # Asegura que todos los índices necesarios existen
                try:
                    mar_value = calculate_mar(face_landmarks, (height, width))
                    self._update_yawn_counter(mar_value)

                    # Para dibujar, usamos los índices de contorno completo
                    mouth_points_to_draw = [face_landmarks[i] for i in config.MOUTH_INDEXES]
                    # Extraer puntos específicos para el MAR para dibujarlos también
                    mouth_points_for_mar_detection = [face_landmarks[i] for i in config.MOUTH_INDEXES_FOR_MAR_CALC]
                    self._draw_mouth_landmarks(frame, mouth_points_to_draw, mouth_points_for_mar_detection) # Llamada modificada

                except Exception as e: # Capturar cualquier otro error durante el cálculo del MAR
                    print(f"Error al calcular MAR: {e}")
                    self.yawn_start_time = None
                    self.detection_reliable = False
            else:
                print("No se detectaron suficientes landmarks para el cálculo del MAR.")
                self.yawn_start_time = None
                self.detection_reliable = False
        else:
            self.yawn_start_time = None
            self.detection_reliable = False
            
        self._draw_info(frame, mar_value)
        
        if not self.detection_reliable:
            self._draw_alert_message(frame)

        return frame, self.yawn_counter

    def _draw_info(self, frame, mar_value):
        """Dibuja el contador de bostezos y el valor MAR en el fotograma."""
        cv2.putText(frame, f"Bostezos: {self.yawn_counter}",
                    config.TEXT_POSITION_YAWN_COUNTER,
                    getattr(cv2, config.FONT),
                    config.FONT_SCALE_INFO,
                    config.COLOR_YAWN_COUNTER,
                    config.FONT_THICKNESS_INFO)
        
        if mar_value >= 0.0 and self.detection_reliable:
            cv2.putText(frame, f"MAR: {mar_value:.2f}",
                        config.TEXT_POSITION_MAR,
                        getattr(cv2, config.FONT),
                        config.FONT_SCALE_INFO,
                        config.COLOR_MAR,
                        config.FONT_THICKNESS_INFO)

    def _draw_mouth_landmarks(self, frame, mouth_points_all, mouth_points_mar): # Modificado
        """Dibuja círculos en los landmarks de la boca."""
        height, width, _ = frame.shape
        # Dibuja los contornos completos
        for point in mouth_points_all:
            x, y = int(point.x * width), int(point.y * height)
            cv2.circle(frame, (x, y), config.LANDMARK_DRAW_RADIUS, config.LANDMARK_DRAW_COLOR, -1)

        # Dibuja los puntos de cálculo del MAR con otro color para distinguirlos
        for point in mouth_points_mar:
            x, y = int(point.x * width), int(point.y * height)
            cv2.circle(frame, (x, y), config.LANDMARK_DRAW_RADIUS, (0, 255, 0), -1) # Verde para MAR
    
    def _draw_alert_message(self, frame):
        """Dibuja un mensaje de alerta en el fotograma."""
        cv2.putText(frame, "Asegúrate de que tu boca esté visible y de frente.",
                    (20, 220), # Ajusta la posición si es necesario
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 255), 2)

    def close(self):
        """Libera los recursos del modelo de MediaPipe."""
        self.face_mesh.close()
