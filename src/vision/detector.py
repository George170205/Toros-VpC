import os
import cv2
import numpy as np
import onnxruntime as ort
from src.domain.vision_models import Frame, FaceDetection
from src.vision.exceptions import DetectionError

def softmax(z):
    assert len(z.shape) == 2
    s = np.max(z, axis=1)
    s = s[:, np.newaxis]
    e_x = np.exp(z - s)
    div = np.sum(e_x, axis=1)
    div = div[:, np.newaxis]
    return e_x / div

def distance2bbox(points, distance, max_shape=None):
    """Decodifica predicciones de distancia a cajas delimitadoras (bboxes)."""
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    if max_shape is not None:
        x1 = np.clip(x1, 0, max_shape[1])
        y1 = np.clip(y1, 0, max_shape[0])
        x2 = np.clip(x2, 0, max_shape[1])
        y2 = np.clip(y2, 0, max_shape[0])
    return np.stack([x1, y1, x2, y2], axis=-1)

def distance2kps(points, distance, max_shape=None):
    """Decodifica predicciones de distancia a landmarks faciales (keypoints)."""
    preds = []
    for i in range(0, distance.shape[1], 2):
        px = points[:, i%2] + distance[:, i]
        py = points[:, i%2+1] + distance[:, i+1]
        if max_shape is not None:
            px = np.clip(px, 0, max_shape[1])
            py = np.clip(py, 0, max_shape[0])
        preds.append(px)
        preds.append(py)
    return np.stack(preds, axis=-1)

class SCRFD:
    """Implementación de inferencia del detector SCRFD usando ONNX Runtime."""
    def __init__(self, model_file: str):
        self.model_file = model_file
        if not os.path.exists(self.model_file):
            raise FileNotFoundError(f"No se encontró el archivo del modelo SCRFD: {self.model_file}")
        
        # Seleccionar Execution Providers (GPU si está disponible)
        providers = ['CPUExecutionProvider']
        if 'CUDAExecutionProvider' in ort.get_available_providers():
            providers = ['CUDAExecutionProvider'] + providers

        self.session = ort.InferenceSession(self.model_file, providers=providers)
        self.center_cache = {}
        self.nms_thresh = 0.4
        self.det_thresh = 0.5
        self._init_vars()

    def _init_vars(self):
        input_cfg = self.session.get_inputs()[0]
        input_shape = input_cfg.shape
        if isinstance(input_shape[2], str):
            self.input_size = None
        else:
            self.input_size = tuple(input_shape[2:4][::-1])
            
        self.input_name = input_cfg.name
        outputs = self.session.get_outputs()
        self.batched = len(outputs[0].shape) == 3
        
        self.output_names = [o.name for o in outputs]
        self.input_mean = 127.5
        self.input_std = 128.0
        self.use_kps = False
        self._num_anchors = 1
        
        # det_10g.onnx de buffalo_l suele tener 9 salidas (scores, bboxes, kpss para stride 8, 16, 32)
        if len(outputs) == 6:
            self.fmc = 3
            self._feat_stride_fpn = [8, 16, 32]
            self._num_anchors = 2
        elif len(outputs) == 9:
            self.fmc = 3
            self._feat_stride_fpn = [8, 16, 32]
            self._num_anchors = 2
            self.use_kps = True
        elif len(outputs) == 10:
            self.fmc = 5
            self._feat_stride_fpn = [8, 16, 32, 64, 128]
            self._num_anchors = 1
        elif len(outputs) == 15:
            self.fmc = 5
            self._feat_stride_fpn = [8, 16, 32, 64, 128]
            self._num_anchors = 1
            self.use_kps = True

    def forward(self, img, threshold):
        scores_list = []
        bboxes_list = []
        kpss_list = []
        input_size = tuple(img.shape[0:2][::-1])
        blob = cv2.dnn.blobFromImage(
            img, 1.0/self.input_std, input_size,
            (self.input_mean, self.input_mean, self.input_mean),
            swapRB=True
        )
        net_outs = self.session.run(self.output_names, {self.input_name: blob})

        input_height = blob.shape[2]
        input_width = blob.shape[3]
        fmc = self.fmc
        
        for idx, stride in enumerate(self._feat_stride_fpn):
            if self.batched:
                scores = net_outs[idx][0]
                bbox_preds = net_outs[idx + fmc][0] * stride
                if self.use_kps:
                    kps_preds = net_outs[idx + fmc * 2][0] * stride
            else:
                scores = net_outs[idx]
                bbox_preds = net_outs[idx + fmc] * stride
                if self.use_kps:
                    kps_preds = net_outs[idx + fmc * 2] * stride

            height = input_height // stride
            width = input_width // stride
            key = (height, width, stride)
            
            if key in self.center_cache:
                anchor_centers = self.center_cache[key]
            else:
                anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
                anchor_centers = (anchor_centers * stride).reshape((-1, 2))
                if self._num_anchors > 1:
                    anchor_centers = np.stack([anchor_centers]*self._num_anchors, axis=1).reshape((-1, 2))
                if len(self.center_cache) < 100:
                    self.center_cache[key] = anchor_centers

            pos_inds = np.where(scores >= threshold)[0]
            bboxes = distance2bbox(anchor_centers, bbox_preds)
            pos_scores = scores[pos_inds]
            pos_bboxes = bboxes[pos_inds]
            scores_list.append(pos_scores)
            bboxes_list.append(pos_bboxes)
            
            if self.use_kps:
                kpss = distance2kps(anchor_centers, kps_preds)
                kpss = kpss.reshape((kpss.shape[0], -1, 2))
                pos_kpss = kpss[pos_inds]
                kpss_list.append(pos_kpss)
                
        return scores_list, bboxes_list, kpss_list

    def detect(self, img, threshold=0.5, input_size=(640, 640)):
        self.det_thresh = threshold
        im_ratio = float(img.shape[0]) / img.shape[1]
        model_ratio = float(input_size[1]) / input_size[0]
        
        if im_ratio > model_ratio:
            new_height = input_size[1]
            new_width = int(new_height / im_ratio)
        else:
            new_width = input_size[0]
            new_height = int(new_width * im_ratio)
            
        det_scale = float(new_height) / img.shape[0]
        resized_img = cv2.resize(img, (new_width, new_height))
        det_img = np.zeros((input_size[1], input_size[0], 3), dtype=np.uint8)
        det_img[:new_height, :new_width, :] = resized_img

        scores_list, bboxes_list, kpss_list = self.forward(det_img, self.det_thresh)

        if len(scores_list) == 0 or sum(score.size for score in scores_list) == 0:
            return np.empty((0, 5), dtype=np.float32), np.empty((0, 5, 2), dtype=np.float32)

        scores = np.vstack(scores_list)
        scores_ravel = scores.ravel()
        order = scores_ravel.argsort()[::-1]
        bboxes = np.vstack(bboxes_list) / det_scale
        
        pre_det = np.hstack((bboxes, scores)).astype(np.float32, copy=False)
        pre_det = pre_det[order, :]
        
        if self.use_kps:
            kpss = np.vstack(kpss_list) / det_scale
            kpss = kpss[order, :, :]
        else:
            kpss = np.empty((pre_det.shape[0], 5, 2), dtype=np.float32)

        # Aplicar NMS
        keep = self.nms(pre_det)
        det = pre_det[keep, :]
        
        if self.use_kps:
            kpss = kpss[keep, :, :]
            
        return det, kpss

    def nms(self, dets):
        thresh = self.nms_thresh
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]

        return keep


class SCRFDFaceDetector:
    """Implementación que cumple con la interfaz FaceDetector de dominio."""
    def __init__(self, model_path: str, default_threshold: float = 0.5):
        self.detector = SCRFD(model_path)
        self.default_threshold = default_threshold

    def detect(self, frame: Frame) -> list[FaceDetection]:
        """Detecta rostros en un frame y devuelve objetos FaceDetection."""
        try:
            # Obtener el tamaño de entrada para inferencia (usando resolución común de 640x640)
            img = frame.image
            bboxes, kpss = self.detector.detect(img, threshold=self.default_threshold, input_size=(640, 640))
            
            detections = []
            for i in range(bboxes.shape[0]):
                bbox = bboxes[i]
                score = float(bbox[4])
                bbox_coords = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
                landmarks = kpss[i] if kpss is not None else np.empty((5, 2))
                
                det_id = f"det_{frame.frame_id}_{i}"
                detections.append(FaceDetection(
                    detection_id=det_id,
                    frame_id=frame.frame_id,
                    bbox=bbox_coords,
                    landmarks=landmarks,
                    confidence=score
                ))
            return detections
        except Exception as e:
            raise DetectionError(f"Error al realizar detección facial en frame {frame.frame_id}: {e}") from e
