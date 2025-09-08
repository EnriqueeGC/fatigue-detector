import cv2
import mediapipe as mp
import time

from utils.earDetector import calculate_ear
import config
from app.controllers import DataController

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
        self.blink_start_time = None
        self.is_eye_closed = False # Nueva variable de estado
        
        self.data_controller = data_controller

    def _update_blink_counter(self, ear_value):
        """
        Actualiza los contadores de parpadeo basado en el valor EAR y el TIEMPO TRANSCURRIDO.
        """
        # --- Lógica de Detección de Parpadeo ---
        if ear_value < config.EAR_THRESHOLD:
            if not self.is_eye_closed:
                self.is_eye_closed = True
                self.blink_start_time = time.time()
                
            # --- Lógica de Alerta de Somnolencia (Nueva) ---
            # Si los ojos llevan cerrados más de 'LONG_BLINK_DURATION_SECONDS'
            if (time.time() - self.blink_start_time) >= config.LONG_BLINK_DURATION_SECONDS:
                # La alerta se dispara y se mantiene mientras los ojos estén cerrados
                print("¡ALERTA DE SOMNOLENCIA! Ojos cerrados por mucho tiempo.")
                
                # Puedes agregar una lógica para no enviar eventos repetidos a la DB cada frame.
                # Por ejemplo, enviar un evento solo una vez por cada período de ojos cerrados.
                # O simplemente, este 'print' es suficiente para una alerta en tiempo real.
        
        # --- Lógica de Conteos (Solo al abrir los ojos) ---
        else: # Si los ojos están abiertos
            if self.is_eye_closed: # Si venían de estar cerrados
                self.is_eye_closed = False
                duration = time.time() - self.blink_start_time

                # Comprobamos si fue un PARPADEO LARGO
                if duration >= config.LONG_BLINK_DURATION_SECONDS:
                    self.long_blink_counter += 1
                    print(f"Parpadeo Largo Finalizado. (Duración: {duration:.2f} s)")
                    
                    # Opcional: Registrar el evento de parpadeo largo en la DB aquí
                    self.data_controller.add_event_to_session(
                        event_type="parpadeo_largo",
                        description=f"Parpadeo largo detectado. Duración: {duration:.2f} s."
                    )

                # Si no fue largo, comprobamos si fue un PARPADEO NORMAL
                elif config.MIN_BLINK_DURATION_SECONDS <= duration <= config.MAX_NORMAL_BLINK_DURATION_SECONDS:
                    self.blink_counter += 1
                    print(f"Parpadeo Normal Detectado (Duración: {duration:.2f} s)")

                self.blink_start_time = None

    def process_frame(self, frame):
        """
        Procesa un único fotograma para detectar parpadeos.
        """
        height, width, _ = frame.shape
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(image_rgb)

        avg_ear = -1.0
        
        if not results.multi_face_landmarks:
            self.blink_start_time = None
            self.is_eye_closed = False # Resetea el estado si no hay cara

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0].landmark

            try:
                left_eye_points = [face_landmarks[i] for i in config.LEFT_EYE_INDEXES]
                right_eye_points = [face_landmarks[i] for i in config.RIGHT_EYE_INDEXES]
                
                left_ear = calculate_ear(left_eye_points, (height, width))
                right_ear = calculate_ear(right_eye_points, (height, width))
                avg_ear = (left_ear + right_ear) / 2.0

                self._update_blink_counter(avg_ear)
                self._draw_eye_landmarks(frame, left_eye_points, right_eye_points)
                
            except IndexError:
                print("No se pudieron detectar ambos ojos. La detección de parpadeo se detiene.")
                self.blink_start_time = None
                self.is_eye_closed = False

        self._draw_info(frame, avg_ear)
        
        # Dibujar la alerta de somnolencia si los ojos están cerrados prolongadamente
        if self.is_eye_closed and (time.time() - self.blink_start_time) >= config.LONG_BLINK_DURATION_SECONDS:
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