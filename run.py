import os
import argparse
import sys
import uvicorn
from src.utils.downloader import ensure_models_exist
from src.utils.config import ConfigurationProvider
from src.application.orchestrator import PipelineOrchestrator
from src.api.server import broadcast_frame_to_websockets, broadcast_event_to_websockets
from src.utils.observability import logger

def main():
    parser = argparse.ArgumentParser(description="Stadium Visitor Recurrent Recognition System")
    parser.add_argument(
        "--camera",
        type=str,
        default="0",
        help="Índice de la webcam (ej. '0') o ruta a un archivo de video .mp4 (ej. 'data/video.mp4')"
    )
    parser.add_argument(
        "--model-det",
        type=str,
        default=os.path.abspath("models/det_10g.onnx"),
        help="Ruta al modelo ONNX de detección facial (Buffalo_L SCRFD)"
    )
    parser.add_argument(
        "--model-rec",
        type=str,
        default=os.path.abspath("models/w600k_r50.onnx"),
        help="Ruta al modelo ONNX de reconocimiento facial (Buffalo_L ArcFace)"
    )
    args = parser.parse_args()

    # 1. Asegurar la existencia de modelos descargándolos si es necesario
    logger.info("Verificando existencia de modelos biométricos...")
    try:
        ensure_models_exist()
    except Exception as e:
        logger.error(f"Fallo al descargar o verificar los modelos: {e}")
        sys.exit(1)

    # 2. Cargar configuración activa
    config = ConfigurationProvider().get_active()
    logger.info(
        "Configuración del sistema cargada.",
        min_quality=config.min_quality,
        min_verification_score=config.min_verification_score
    )

    # 3. Determinar el origen de cámara (convertir a entero si es un dígito)
    camera_source = args.camera
    if camera_source.isdigit():
        camera_source = int(camera_source)
        logger.info(f"Usando Webcam local con índice: {camera_source}")
    else:
        if not os.path.exists(camera_source):
            logger.error(f"Archivo de video de prueba especificado no existe: {camera_source}")
            sys.exit(1)
        logger.info(f"Usando archivo de video de prueba: {camera_source}")

    # 4. Instanciar el Coordinador/Orquestador del pipeline
    orchestrator = PipelineOrchestrator(
        camera_id=str(camera_source),
        model_det_path=args.model_det,
        model_rec_path=args.model_rec,
        config=config,
        live_event_callback=broadcast_event_to_websockets,
        frame_callback=broadcast_frame_to_websockets
    )

    # 5. Iniciar orquestador en segundo plano
    logger.info("Iniciando Pipeline de Visión y Biometría...")
    orchestrator.start()

    # 6. Lanzar servidor FastAPI
    logger.info("Iniciando API Backend en puerto 8000...")
    try:
        uvicorn.run("src.api.server:app", host="0.0.0.0", port=8000, log_level="info")
    except KeyboardInterrupt:
        logger.info("Interrupción por teclado detectada.")
    finally:
        logger.info("Deteniendo Pipeline de forma segura...")
        orchestrator.stop()
        logger.info("Shutdown completo.")

if __name__ == "__main__":
    main()
