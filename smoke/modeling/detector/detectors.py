from .keypoint_detector import KeypointDetector

def build_detection_model(cfg):
    return KeypointDetector(cfg)