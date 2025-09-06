from .models import Session, Event, SessionLocal
import datetime

class DataController:
    """Clase controladora para manejar las operaciones de la base de datos."""

    def __init__(self):
        """Inicializa la conexión con la base de datos."""
        self.db = SessionLocal()
        self.current_session = None

    def start_new_session(self):
        """Crea una nueva sesión en la base de datos al inicio del script."""
        try:
            self.current_session = Session()
            self.db.add(self.current_session)
            self.db.commit()
            print(f"Nueva sesión iniciada con ID: {self.current_session.id}")
            return self.current_session.id
        except Exception as e:
            self.db.rollback()
            print(f"Error al iniciar sesión: {e}")
            return None

    def add_event_to_session(self, event_type: str, description: str):
        """Agrega un evento a la sesión actual si está activa."""
        if self.current_session is None:
            print("No hay una sesión activa para añadir el evento.")
            return

        try:
            new_event = Event(
                event_type=event_type,
                description=description,
                session_id=self.current_session.id
            )
            self.db.add(new_event)
            self.db.commit()
            print(f"Evento '{event_type}' añadido a la sesión {self.current_session.id}.")
        except Exception as e:
            self.db.rollback()
            print(f"Error al añadir evento: {e}")

    def end_current_session(self):
        """Registra el tiempo de finalización de la sesión actual."""
        if self.current_session is None:
            print("No hay una sesión activa para finalizar.")
            return

        try:
            self.current_session.end_time = datetime.datetime.now()
            self.db.commit()
            print(f"Sesión {self.current_session.id} finalizada.")
        except Exception as e:
            self.db.rollback()
            print(f"Error al finalizar sesión: {e}")
        finally:
            self.db.close()
