import logging
import json
import os
import sys
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Any

# Crear carpeta de logs si no existe
LOGS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "logs"))
os.makedirs(LOGS_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOGS_DIR, "system.log")

# Configurar el logger básico para consola y archivo
logger_backend = logging.getLogger("VisitorSystem")
logger_backend.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(message)s')

# Handler para archivo
file_handler = logging.FileHandler(LOG_FILE, encoding='utf-8')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
logger_backend.addHandler(file_handler)

# Handler para consola
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s', datefmt='%H:%M:%S'))
logger_backend.addHandler(console_handler)


@dataclass(frozen=True, slots=True)
class CorrelationContext:
    camera_id: str | None = None
    session_id: str | None = None
    track_id: int | None = None
    task_id: str | None = None
    evidence_id: str | None = None
    decision_id: str | None = None
    event_id: str | None = None


class StructuredLogger:
    def __init__(self, name: str = "VisitorSystem"):
        self.logger = logger_backend
        self._context = CorrelationContext()

    def with_context(self, context: CorrelationContext) -> 'StructuredLogger':
        """Crea una instancia con el contexto de correlación inyectado."""
        new_logger = StructuredLogger()
        new_logger._context = context
        return new_logger

    def _format_message(self, level: str, event: str, fields: dict[str, Any]) -> str:
        ctx_dict = {k: v for k, v in asdict(self._context).items() if v is not None}
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": level,
            "event": event,
            "context": ctx_dict,
            **fields
        }
        return json.dumps(log_data, ensure_ascii=False)

    def debug(self, event: str, **fields) -> None:
        self.logger.debug(self._format_message("DEBUG", event, fields))

    def info(self, event: str, **fields) -> None:
        self.logger.info(self._format_message("INFO", event, fields))

    def warning(self, event: str, **fields) -> None:
        self.logger.warning(self._format_message("WARNING", event, fields))

    def error(self, event: str, **fields) -> None:
        self.logger.error(self._format_message("ERROR", event, fields))

    def exception(self, event: str, **fields) -> None:
        import traceback
        fields["exception"] = traceback.format_exc()
        self.logger.error(self._format_message("EXCEPTION", event, fields))


class MetricsCollector:
    def __init__(self):
        self._counters: dict[str, int] = {}
        self._gauges: dict[str, float] = {}
        self._observations: dict[str, list[float]] = {}

    def increment(self, name: str, value: int = 1, **labels) -> None:
        label_str = json.dumps(labels, sort_keys=True) if labels else ""
        key = f"{name}{label_str}"
        self._counters[key] = self._counters.get(key, 0) + value

    def gauge(self, name: str, value: float, **labels) -> None:
        label_str = json.dumps(labels, sort_keys=True) if labels else ""
        key = f"{name}{label_str}"
        self._gauges[key] = value

    def observe(self, name: str, value: float, **labels) -> None:
        label_str = json.dumps(labels, sort_keys=True) if labels else ""
        key = f"{name}{label_str}"
        if key not in self._observations:
            self._observations[key] = []
        self._observations[key].append(value)

    def get_metrics_snapshot(self) -> dict[str, Any]:
        """Devuelve una instantánea de las métricas recopiladas."""
        summary_obs = {}
        for key, vals in self._observations.items():
            if vals:
                summary_obs[key] = {
                    "count": len(vals),
                    "mean": sum(vals) / len(vals),
                    "min": min(vals),
                    "max": max(vals)
                }
        return {
            "counters": self._counters.copy(),
            "gauges": self._gauges.copy(),
            "observations": summary_obs
        }
        
# Instancias globales
logger = StructuredLogger()
metrics = MetricsCollector()
