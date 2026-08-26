import json
import os
from datetime import datetime
from src.domain.persistence_models import SystemConfiguration

CONFIG_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "config.json"))

DEFAULT_CONFIG = SystemConfiguration(
    configuration_id="config_default_v1",
    version="1.0.0",
    min_quality=0.45,
    top_k=5,
    candidate_delta=0.08,
    min_verification_score=0.68,
    min_margin=0.08,
    min_consistency=0.72,
    max_secondary_embeddings=5,
    primary_update_lambda=0.85,
    created_at=datetime.now(),
    active=True
)

class ConfigurationProvider:
    def __init__(self):
        self._current_config = DEFAULT_CONFIG
        self._load_local_config()

    def _load_local_config(self):
        """Intenta cargar la configuración desde un archivo config.json local."""
        try:
            if os.path.exists(CONFIG_FILE):
                with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    
                # Parsear fecha
                created_at = datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.now()
                
                self._current_config = SystemConfiguration(
                    configuration_id=data.get("configuration_id", DEFAULT_CONFIG.configuration_id),
                    version=data.get("version", DEFAULT_CONFIG.version),
                    min_quality=float(data.get("min_quality", DEFAULT_CONFIG.min_quality)),
                    top_k=int(data.get("top_k", DEFAULT_CONFIG.top_k)),
                    candidate_delta=float(data.get("candidate_delta", DEFAULT_CONFIG.candidate_delta)),
                    min_verification_score=float(data.get("min_verification_score", DEFAULT_CONFIG.min_verification_score)),
                    min_margin=float(data.get("min_margin", DEFAULT_CONFIG.min_margin)),
                    min_consistency=float(data.get("min_consistency", DEFAULT_CONFIG.min_consistency)),
                    max_secondary_embeddings=int(data.get("max_secondary_embeddings", DEFAULT_CONFIG.max_secondary_embeddings)),
                    primary_update_lambda=float(data.get("primary_update_lambda", DEFAULT_CONFIG.primary_update_lambda)),
                    created_at=created_at,
                    active=bool(data.get("active", DEFAULT_CONFIG.active))
                )
        except Exception as e:
            print(f"Error al cargar configuración local, usando valores por defecto: {e}")

    def save_config(self, config: SystemConfiguration):
        """Guarda la configuración en el archivo local."""
        try:
            os.makedirs(os.path.dirname(CONFIG_FILE), exist_ok=True)
            data = {
                "configuration_id": config.configuration_id,
                "version": config.version,
                "min_quality": config.min_quality,
                "top_k": config.top_k,
                "candidate_delta": config.candidate_delta,
                "min_verification_score": config.min_verification_score,
                "min_margin": config.min_margin,
                "min_consistency": config.min_consistency,
                "max_secondary_embeddings": config.max_secondary_embeddings,
                "primary_update_lambda": config.primary_update_lambda,
                "created_at": config.created_at.isoformat(),
                "active": config.active
            }
            with open(CONFIG_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            self._current_config = config
        except Exception as e:
            print(f"Error al guardar configuración local: {e}")

    def get_active(self) -> SystemConfiguration:
        return self._current_config
