import cv2
import mediapipe as mp
import time
import collections

from utils.earDetector import calculate_ear
import config
from app.controllers import DataController

from utils.beepAlert import beep_alerta

class BlinkDetector:
    def __init__(self, data_controller: DataController):
        """Inicializa el detector con el modelo de MediaPipe y el controlador de DB."""
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=config.MAX_FACES,
            refine_landmarks=True,
            min_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE
        )
        
        # --- Variables de estado del detector ---
        self.blink_counter = 0
        self.long_blink_counter = 0
        self.blink_start_time = None
        self.is_eye_closed = False
        self.data_controller = data_controller

        # --- Variables para el suavizado del EAR (Media Móvil) ---
        self.ear_history = collections.deque(maxlen=config.SMOOTHING_FRAMES)
        self.smoothed_ear = -1.0

        # --- Variables para el umbral adaptativo ---
        self.is_calibrating = True
        self.calibration_start_time = time.time()
        self.ear_calibration_values = []
        self.adaptive_ear_threshold = config.EAR_THRESHOLD # Valor por defecto hasta la calibración
        
        # Nuevos umbrales para mejorar la lógica
        self.open_ear_base = None
        self.closed_ear_threshold = None
        self.drowsy_ear_threshold = None

    def _calibrate_threshold(self, ear_value):
        """Fase de calibración para definir el umbral adaptativo."""
        current_time = time.time()
        if current_time - self.calibration_start_time < config.CALIBRATION_DURATION_SECONDS:
            # Sigue en fase de calibración
            self.ear_calibration_values.append(ear_value)
        else:
            # Termina la calibración y calcula el umbral
            if self.ear_calibration_values:
                # Filtrar valores extremos (posibles parpadeos)
                valid_ears = [ear for ear in self.ear_calibration_values if ear > 0]
                if valid_ears:
                    # Usamos el percentil 95 para evitar que parpadeos cortos afecten el promedio
                    self.open_ear_base = sorted(valid_ears)[int(len(valid_ears) * 0.95)]
                    
                    # Definimos los umbrales basados en el valor base
                    self.closed_ear_threshold = self.open_ear_base * config.CLOSED_EYE_RATIO
                    self.drowsy_ear_threshold = self.open_ear_base * config.DROWSY_RATIO
                    
                    self.adaptive_ear_threshold = self.closed_ear_threshold
                    print(f"Calibración finalizada. EAR base: {self.open_ear_base:.2f}, Umbral de cierre: {self.closed_ear_threshold:.2f}")
                else:
                    print("Calibración fallida: No se detectaron valores de EAR válidos.")
            
            self.is_calibrating = False

    def _update_blink_counter(self, ear_value):
        """
        Actualiza los contadores de parpadeo basado en el valor EAR y el tiempo.
        """
        # Se usa el umbral adaptativo
        if ear_value < self.adaptive_ear_threshold:
            if not self.is_eye_closed:
                self.is_eye_closed = True
                self.blink_start_time = time.time()
                
            # Lógica de Alerta de Somnolencia
            if self.blink_start_time and (time.time() - self.blink_start_time) >= config.LONG_BLINK_DURATION_SECONDS:
                print("¡ALERTA DE SOMNOLENCIA! Ojos cerrados por mucho tiempo.")
                beep_alerta()
        else:
            if self.is_eye_closed:
                self.is_eye_closed = False
                duration = time.time() - self.blink_start_time

                if duration >= config.LONG_BLINK_DURATION_SECONDS:
                    self.long_blink_counter += 1
                    print(f"Parpadeo Largo Finalizado. (Duración: {duration:.2f} s)")
                    self.data_controller.add_event_to_session(
                        event_type="parpadeo_largo",
                        description=f"Parpadeo largo detectado. Duración: {duration:.2f} s."
                    )
                elif config.MIN_BLINK_DURATION_SECONDS <= duration <= config.MAX_NORMAL_BLINK_DURATION_SECONDS:
                    self.blink_counter += 1
                self.blink_start_time = None

    def process_frame(self, frame):
        """
        Procesa un único fotograma para detectar parpadeos, con lógica mejorada
        para manejar la ausencia de la cara y el suavizado del EAR.
        """
        height, width, _ = frame.shape
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(image_rgb)
        
        avg_ear = -1.0
        
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0].landmark
            try:
                left_eye_points = [face_landmarks[i] for i in config.LEFT_EYE_INDEXES]
                right_eye_points = [face_landmarks[i] for i in config.RIGHT_EYE_INDEXES]
                
                if not any(p.y > 0.8 for p in left_eye_points + right_eye_points):
                    left_ear = calculate_ear(left_eye_points, (height, width))
                    right_ear = calculate_ear(right_eye_points, (height, width))
                    avg_ear = (left_ear + right_ear) / 2.0
            except IndexError:
                avg_ear = -1.0
        
        # Lógica de suavizado del EAR
        if avg_ear >= 0.0:
            self.ear_history.append(avg_ear)
            self.smoothed_ear = sum(self.ear_history) / len(self.ear_history)
        else:
            self.ear_history.clear()
            self.smoothed_ear = -1.0

        # Lógica de calibración o detección
        if self.is_calibrating:
            if self.smoothed_ear >= 0.0:
                self._calibrate_threshold(self.smoothed_ear)
        else:
            self._update_blink_counter(self.smoothed_ear)
        
        # Dibujado de landmarks, info y alertas
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0].landmark
            left_eye_points = [face_landmarks[i] for i in config.LEFT_EYE_INDEXES]
            right_eye_points = [face_landmarks[i] for i in config.RIGHT_EYE_INDEXES]
            self._draw_eye_landmarks(frame, left_eye_points, right_eye_points)
        
        self._draw_info(frame, self.smoothed_ear)
        
        if self.is_calibrating:
            cv2.putText(frame, "Calibrando... Mantenga los ojos abiertos", (50, 250),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        elif self.is_eye_closed and self.blink_start_time and (time.time() - self.blink_start_time) >= config.LONG_BLINK_DURATION_SECONDS:
            cv2.putText(frame, "¡ALERTA DE SOMNOLENCIA!", (50, 250), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

        return frame, self.blink_counter, self.long_blink_counter

    def _draw_info(self, frame, ear_value):
        """Dibuja el contador de parpadeos y el valor EAR en el fotograma."""
        # La lógica de dibujado se mantiene igual para evitar redundancia en el ejemplo
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