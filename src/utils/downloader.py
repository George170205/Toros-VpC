import os
import requests
from tqdm import tqdm

MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "models"))

MODEL_URLS = {
    "det_10g.onnx": "https://huggingface.co/yolkailtd/face-swap-models/resolve/main/insightface/models/buffalo_l/det_10g.onnx",
    "w600k_r50.onnx": "https://huggingface.co/yolkailtd/face-swap-models/resolve/main/insightface/models/buffalo_l/w600k_r50.onnx"
}

def download_file(url: str, dest_path: str):
    """Descarga un archivo desde una URL mostrando una barra de progreso."""
    temp_dest = dest_path + ".tmp"
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024 * 1024  # 1 MB
        
        desc = os.path.basename(dest_path)
        with open(temp_dest, 'wb') as file, tqdm(
            desc=desc,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(block_size):
                size = file.write(data)
                bar.update(size)
        
        # Renombrar al terminar con éxito
        if os.path.exists(dest_path):
            os.remove(dest_path)
        os.rename(temp_dest, dest_path)
        print(f"Descargado con éxito: {desc}")
    except Exception as e:
        if os.path.exists(temp_dest):
            os.remove(temp_dest)
        print(f"Error al descargar {os.path.basename(dest_path)}: {e}")
        raise e

def ensure_models_exist():
    """Verifica la existencia de los modelos ONNX y los descarga si faltan."""
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
        print(f"Creado directorio de modelos: {MODELS_DIR}")
        
    for filename, url in MODEL_URLS.items():
        dest_path = os.path.join(MODELS_DIR, filename)
        if not os.path.exists(dest_path):
            print(f"Modelo no encontrado: {filename}. Iniciando descarga automática...")
            download_file(url, dest_path)
        else:
            print(f"Modelo verificado: {filename}")

if __name__ == "__main__":
    ensure_models_exist()
