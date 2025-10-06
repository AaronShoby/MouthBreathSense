import os
import yaml
import numpy as np
import onnxruntime as ort


def _project_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def _load_config():
    cfg_path = os.path.join(_project_root(), 'config.yaml')
    print(f"[inference] Loading config from {cfg_path}")
    with open(cfg_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def predict(sample):
    """Run a single prediction using the exported ONNX model."""
    cfg = _load_config()
    model_path = os.path.join(_project_root(), cfg.get('model_path', 'models/model.onnx'))
    print(f"[inference] Loading ONNX model at {model_path}")
    sess = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])

    # Prepare input: expect a single-channel 28x28 sample
    arr = np.array(sample, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[None, None, :, :]  # (1,1,H,W)
    elif arr.ndim == 3:
        arr = arr[None, :, :, :]     # (1,C,H,W)
    elif arr.ndim == 4:
        pass
    else:
        # Fall back to a dummy input if shape is unexpected
        print("[inference] Unexpected input shape; using dummy 28x28.")
        arr = np.random.randn(1, 1, 28, 28).astype(np.float32)

    input_name = sess.get_inputs()[0].name
    outputs = sess.run(None, {input_name: arr})[0]
    pred = int(np.argmax(outputs, axis=1)[0])
    print(f"[inference] Predicted class: {pred}")
    return pred


if __name__ == '__main__':
    dummy = np.random.randn(28, 28).astype(np.float32)
    predict(dummy)
