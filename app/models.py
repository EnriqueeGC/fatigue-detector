import datetime
from sqlalchemy import create_engine, Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

# Configuración de la base de datos
DATABASE_URL = "sqlite:///database.db"
engine = create_engine(DATABASE_URL)
Base = declarative_base()

# Modelo para la tabla de Sesiones
class Session(Base):
    __tablename__ = "sessions"
    id = Column(Integer, primary_key=True, autoincrement=True)
    start_time = Column(DateTime, default=datetime.datetime.now)
    end_time = Column(DateTime, nullable=True)
    events = relationship("Event", back_populates="session", cascade="all, delete-orphan")

# Modelo para la tabla de Eventos
class Event(Base):
    __tablename__ = "events"
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.datetime.now)
    event_type = Column(String)  # Ej. 'distraccion', 'fatiga'
    description = Column(String)
    session_id = Column(Integer, ForeignKey("sessions.id"))
    session = relationship("Session", back_populates="events")

# Crear las tablas en la base de datos si no existen
Base.metadata.create_all(engine)

# Configurar la sesión de SQLAlchemy
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)