import cv2
import atexit

from modules.blinkDetector import BlinkDetector
from modules.yawnDetector import YawnDetector # ¡Importamos el nuevo detector!
from app.controllers import DataController

# Creamos una instancia del DataController al inicio del script
data_controller = DataController()

def main():
    # Inicializar la cámara
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: No se puede abrir la cámara.")
        return

    # 1. Iniciar una nueva sesión de base de datos
    data_controller.start_new_session()
    
    # 2. Crear instancias de ambos detectores, ¡pasándoles el controlador!
    blink_detector = BlinkDetector(data_controller)
    yawn_detector = YawnDetector(data_controller) # Instancia del nuevo detector

    # Bucle principal de procesamiento
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Fin del stream de video o error al leer el fotograma.")
            break

        frame = cv2.flip(frame, 1)

        # 3. Procesar el fotograma con AMBOS detectores
        
        # Procesar con el detector de parpadeos
        processed_frame, normal_blink_count, long_blink_count = blink_detector.process_frame(frame)
        
        # El fotograma procesado por el detector de parpadeos se pasa al detector de bostezos
        # Esto asegura que todas las anotaciones (texto y círculos) se dibujen en el mismo fotograma
        final_frame, yawn_count = yawn_detector.process_frame(processed_frame)

        # 4. Actualizar la visualización de los contadores en el fotograma final
        cv2.putText(final_frame, f"Parpadeos Normales: {normal_blink_count}",
                    (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(final_frame, f"Parpadeos Largos: {long_blink_count}",
                    (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(final_frame, f"Bostezos: {yawn_count}",
                    (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow('Deteccion de Fatiga y Somnolencia', final_frame)

        if cv2.waitKey(5) & 0xFF == 27:
            break

    # Limpieza final
    blink_detector.close()
    yawn_detector.close() # Cierre del nuevo detector
    cap.release()
    cv2.destroyAllWindows()
    
    # 5. Finalizar la sesión de base de datos de forma segura
    data_controller.end_current_session()
    print("Programa finalizado y sesión de base de datos cerrada.")

if __name__ == '__main__':
    main()