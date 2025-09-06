"""
Módulo que contiene la clase BlinkDetector, responsable de procesar
los fotogramas de video para detectar y contar parpadeos.
"""

import cv2
import mediapipe as mp
import time

from utils.earDetector import calculate_ear
import config
from app.controllers import DataController  # ¡Importamos el controlador aquí!

class BlinkDetector:
    def __init__(self, data_controller: DataController):
        """Inicializa el detector con el modelo de MediaPipe y el controlador de DB."""
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=config.MAX_FACES,
            refine_landmarks=True,
            min_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE
        )
        
        # --- Variables de estado actualizadas ---
        self.blink_counter = 0
        self.long_blink_counter = 0
        self.blink_start_time = None  # <--- Reemplaza a frame_counter y is_eye_closed
        
        self.data_controller = data_controller

    def _update_blink_counter(self, ear_value):
        """
        Actualiza los contadores de parpadeo basado en el valor EAR y el TIEMPO TRANSCURRIDO.
        """
        # --- Lógica robusta basada en tiempo ---

        # Si el ojo está cerrado (EAR por debajo del umbral)
        if ear_value < config.EAR_THRESHOLD:
            # Si es la primera vez que detectamos el ojo cerrado, guardamos el tiempo de inicio.
            if self.blink_start_time is None:
                self.blink_start_time = time.time()
        
        # Si el ojo está abierto
        else:
            # Y si venía de estar cerrado (tenemos un tiempo de inicio guardado)
            if self.blink_start_time is not None:
                # Calculamos la duración total que el ojo estuvo cerrado en segundos.
                duration = time.time() - self.blink_start_time

                # 1. Comprobamos si fue un PARPADEO LARGO (fatiga)
                if duration >= config.LONG_BLINK_DURATION_SECONDS:
                    self.long_blink_counter += 1
                    print(f"¡Parpadeo Largo Detectado! (Duración: {duration:.2f} segundos)")
                    
                    # Registramos el evento de fatiga en la base de datos
                    self.data_controller.add_event_to_session(
                        event_type="fatiga",
                        description=f"Parpadeo largo detectado. Duración: {duration:.2f} s."
                    )

                # 2. Si no fue largo, comprobamos si fue un PARPADEO NORMAL
                elif config.MIN_BLINK_DURATION_SECONDS <= duration <= config.MAX_NORMAL_BLINK_DURATION_SECONDS:
                    self.blink_counter += 1
                    print(f"Parpadeo Normal Detectado (Duración: {duration:.2f} segundos)")

                # 3. Finalmente, reseteamos el tiempo de inicio para estar listos para el próximo parpadeo.
                self.blink_start_time = None

    def process_frame(self, frame):
        """
        Procesa un único fotograma para detectar parpadeos.
        """
        height, width, _ = frame.shape
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(image_rgb)

        avg_ear = -1.0

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0].landmark

            left_eye_points = [face_landmarks[i] for i in config.LEFT_EYE_INDEXES]
            right_eye_points = [face_landmarks[i] for i in config.RIGHT_EYE_INDEXES]

            left_ear = calculate_ear(left_eye_points, (height, width))
            right_ear = calculate_ear(right_eye_points, (height, width))
            avg_ear = (left_ear + right_ear) / 2.0

            self._update_blink_counter(avg_ear)
            self._draw_eye_landmarks(frame, left_eye_points, right_eye_points)

        self._draw_info(frame, avg_ear)

        return frame, self.blink_counter, self.long_blink_counter

    def _draw_info(self, frame, ear_value):
        """Dibuja el contador de parpadeos y el valor EAR en el fotograma."""
        cv2.putText(frame, f"Parpadeos: {self.blink_counter}",
                    config.TEXT_POSITION_COUNTER,
                    getattr(cv2, config.FONT),
                    config.FONT_SCALE_INFO,
                    config.COLOR_COUNTER,
                    config.FONT_THICKNESS_INFO)
        
        if ear_value >= 0.0:
            cv2.putText(frame, f"EAR: {ear_value:.2f}",
                        config.TEXT_POSITION_EAR,
                        getattr(cv2, config.FONT),
                        config.FONT_SCALE_INFO,
                        config.COLOR_EAR,
                        config.FONT_THICKNESS_INFO)
                        
    def _draw_eye_landmarks(self, frame, left_eye, right_eye):
        """Dibuja círculos en los landmarks de los ojos."""
        height, width, _ = frame.shape
        for point in left_eye + right_eye:
            x, y = int(point.x * width), int(point.y * height)
            cv2.circle(frame, (x, y), config.LANDMARK_DRAW_RADIUS, config.LANDMARK_DRAW_COLOR, -1)

    def close(self):
        """Libera los recursos del modelo de MediaPipe."""
        self.face_mesh.close()