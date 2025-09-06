import cv2
import atexit # Módulo para asegurar la ejecución de una función al salir

from modules.blinkDetector import BlinkDetector
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
    
    # 2. Crear una instancia de nuestro detector, ¡pasándole el controlador!
    detector = BlinkDetector(data_controller)

    # Bucle principal de procesamiento
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Fin del stream de video o error al leer el fotograma.")
            break

        frame = cv2.flip(frame, 1)

        # Procesar el fotograma y obtener ambos contadores
        processed_frame, normal_blink_count, long_blink_count = detector.process_frame(frame)

        cv2.putText(processed_frame, f"Parpadeos Normales: {normal_blink_count}",
                    (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(processed_frame, f"Parpadeos Largos: {long_blink_count}",
                    (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow('Deteccion de Parpadeos - Somnolencia', processed_frame)

        if cv2.waitKey(5) & 0xFF == 27:
            break

    # Limpieza final
    detector.close()
    cap.release()
    cv2.destroyAllWindows()
    
    # 3. Finalizar la sesión de base de datos de forma segura
    data_controller.end_current_session()
    print("Programa finalizado y sesión de base de datos cerrada.")

if __name__ == '__main__':
    main()