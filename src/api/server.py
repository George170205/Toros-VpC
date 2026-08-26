import asyncio
import base64
import os
import cv2
import json
import threading
from datetime import datetime
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any
from src.domain import SystemConfiguration, PersonStatus, Camera
from src.persistence import DatabaseService, UnitOfWork
from src.utils.observability import logger, metrics

# Configurar FastAPI
app = FastAPI(title="Stadium Visitor Recognition System API", version="1.0.0")

# Permitir CORS para desarrollo local con React/Vite
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Conexiones activas de WebSocket
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.lock = threading.Lock() if 'threading' in globals() else None

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast_json(self, message: Dict[str, Any]):
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                disconnected.append(connection)
        
        for conn in disconnected:
            self.disconnect(conn)

manager = ConnectionManager()
db_service = DatabaseService()

# Canal para enviar frames anotados desde el orquestador hacia los WebSockets
frame_loop = None

def broadcast_frame_to_websockets(frame_data: str):
    """Encola el frame en el loop de eventos asíncronos para transmitir por WebSocket."""
    if frame_loop:
        asyncio.run_coroutine_threadsafe(
            manager.broadcast_json({"type": "frame", "data": frame_data}),
            frame_loop
        )

def broadcast_event_to_websockets(event_data: dict):
    """Encola una decisión biométrica en el loop de eventos asíncronos para transmitir por WebSocket."""
    if frame_loop:
        asyncio.run_coroutine_threadsafe(
            manager.broadcast_json({"type": "event", "data": event_data}),
            frame_loop
        )


@app.on_event("startup")
async def startup_event():
    global frame_loop
    frame_loop = asyncio.get_event_loop()
    logger.info("API Backend Iniciada. Loop de eventos enlazado.")


@app.websocket("/api/ws/stream")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Mantener conexión recibiendo pings
            data = await websocket.receive_text()
            # Si el cliente envía 'ping', respondemos 'pong'
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        manager.disconnect(websocket)


@app.get("/api/visits")
def get_visits(limit: int = 50):
    """Obtiene el historial de visitas registradas."""
    try:
        with UnitOfWork(db_service) as uow:
            # Query directa simplificada
            cursor = uow.conn.execute(
                """
                SELECT v.*, p.public_code, p.total_visits 
                FROM visits v
                JOIN persons p ON v.person_id = p.person_id
                ORDER BY v.entry_timestamp DESC LIMIT ?
                """, (limit,)
            )
            rows = cursor.fetchall()
            visits = []
            for r in rows:
                visits.append({
                    "visit_id": r["visit_id"],
                    "event_id": r["event_id"],
                    "person_id": r["person_id"],
                    "public_code": r["public_code"],
                    "camera_id": r["camera_id"],
                    "timestamp": r["entry_timestamp"],
                    "confidence": r["confidence"],
                    "entry_zone": r["entry_zone"],
                    "total_visits": r["total_visits"]
                })
            return visits
    except Exception as e:
        logger.error("api.get_visits_failed", error=str(e))
        return []


@app.get("/api/persons")
def get_persons(limit: int = 50):
    """Obtiene la lista de personas registradas."""
    try:
        with UnitOfWork(db_service) as uow:
            cursor = uow.conn.execute(
                """
                SELECT p.*, e.quality_score as primary_quality 
                FROM persons p
                LEFT JOIN embeddings e ON p.primary_embedding_id = e.embedding_id
                ORDER BY p.last_seen DESC LIMIT ?
                """, (limit,)
            )
            rows = cursor.fetchall()
            persons = []
            for r in rows:
                persons.append({
                    "person_id": r["person_id"],
                    "public_code": r["public_code"],
                    "status": r["status"],
                    "first_seen": r["first_seen"],
                    "last_seen": r["last_seen"],
                    "total_visits": r["total_visits"],
                    "primary_quality": r["primary_quality"]
                })
            return persons
    except Exception as e:
        logger.error("api.get_persons_failed", error=str(e))
        return []


@app.get("/api/config")
def get_config():
    """Obtiene la configuración activa."""
    try:
        with UnitOfWork(db_service) as uow:
            config = uow.conn.execute(
                "SELECT * FROM system_configurations WHERE active = 1 LIMIT 1"
            ).fetchone()
            if config:
                params = json.loads(config["parameters_json"])
                return {
                    "configuration_id": config["configuration_id"],
                    "version": config["version"],
                    "created_at": config["created_at"],
                    **params
                }
            return {}
    except Exception as e:
        logger.error("api.get_config_failed", error=str(e))
        return {}


@app.post("/api/config")
def update_config(new_params: Dict[str, Any]):
    """Actualiza los parámetros de configuración."""
    try:
        with UnitOfWork(db_service) as uow:
            # Obtener configuración activa actual
            row = uow.conn.execute(
                "SELECT * FROM system_configurations WHERE active = 1 LIMIT 1"
            ).fetchone()
            
            if row:
                current_params = json.loads(row["parameters_json"])
                # Fusionar con los nuevos parámetros
                current_params.update(new_params)
                
                # Desactivar la configuración anterior e insertar la nueva versión
                new_version = f"config_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                
                uow.conn.execute(
                    "UPDATE system_configurations SET active = 0 WHERE configuration_id = ?",
                    (row["configuration_id"],)
                )
                
                uow.conn.execute(
                    """
                    INSERT INTO system_configurations (configuration_id, version, parameters_json, created_at, active)
                    VALUES (?, ?, ?, ?, 1)
                    """,
                    (new_version, "1.0.0", json.dumps(current_params), datetime.now().isoformat())
                )
                logger.info("api.config_updated", new_version=new_version)
                return {"status": "success", "configuration_id": new_version}
            return {"status": "error", "message": "No active configuration found"}
    except Exception as e:
        logger.error("api.update_config_failed", error=str(e))
        return {"status": "error", "message": str(e)}


@app.get("/api/metrics")
def get_metrics():
    """Obtiene una instantánea de las métricas de telemetría de observabilidad."""
    return metrics.get_metrics_snapshot()


@app.post("/api/reset")
def reset_database():
    """Borra todos los datos (útil para pruebas limpias de enrolamiento)."""
    try:
        with UnitOfWork(db_service) as uow:
            uow.conn.execute("DELETE FROM visits;")
            uow.conn.execute("DELETE FROM embeddings;")
            uow.conn.execute("DELETE FROM persons;")
            uow.conn.execute("DELETE FROM track_session_logs;")
            uow.conn.execute("DELETE FROM identification_decisions;")
            logger.info("api.database_reset")
            return {"status": "success"}
    except Exception as e:
        logger.error("api.reset_database_failed", error=str(e))
        return {"status": "error", "message": str(e)}
