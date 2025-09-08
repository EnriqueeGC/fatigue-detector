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
            
            try:
                # Pasamos TODOS los face_landmarks a calculate_mar
                mar_value = calculate_mar(face_landmarks, (height, width))
                self._update_yawn_counter(mar_value)
                
                # Para dibujar, usamos los índices de contorno completo
                mouth_points_to_draw = [face_landmarks[i] for i in config.MOUTH_INDEXES]
                self._draw_mouth_landmarks(frame, mouth_points_to_draw)
                
            except IndexError:
                print("No se pudieron detectar los landmarks de la boca.")
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

    def _draw_mouth_landmarks(self, frame, mouth_points):
        """Dibuja círculos en los landmarks de la boca."""
        height, width, _ = frame.shape
        for point in mouth_points:
            x, y = int(point.x * width), int(point.y * height)
            cv2.circle(frame, (x, y), config.LANDMARK_DRAW_RADIUS, config.LANDMARK_DRAW_COLOR, -1)
    
    def _draw_alert_message(self, frame):
        """Dibuja un mensaje de alerta en el fotograma."""
        cv2.putText(frame, "Asegúrate de que tu boca esté visible y de frente.",
                    (20, 220), # Ajusta la posición si es necesario
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 255), 2)

    def close(self):
        """Libera los recursos del modelo de MediaPipe."""
        self.face_mesh.close()
