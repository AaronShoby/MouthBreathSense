import os
import yaml
import numpy as np
import onnxruntime as ort
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

from src.dataset import load_dataset
from src.utils import load_model


def _project_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def _load_config():
    cfg_path = os.path.join(_project_root(), 'config.yaml')
    print(f"[evaluate] Loading config from {cfg_path}")
    with open(cfg_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def evaluate():
    cfg = _load_config()
    model_path = os.path.join(_project_root(), cfg.get('model_path', 'models/model.onnx'))
    batch_size = int(cfg.get('batch_size', 16))

    print(f"[evaluate] Creating test data loader (batch_size={batch_size})...")
    _, test_loader = load_dataset(batch_size=batch_size)

    print(f"[evaluate] Loading ONNX model: {model_path}")
    sess = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    input_name = sess.get_inputs()[0].name

    y_true = []
    y_pred = []

    for inputs, targets in test_loader:
        np_inp = inputs.numpy().astype(np.float32)
        outputs = sess.run(None, {input_name: np_inp})[0]
        preds = np.argmax(outputs, axis=1)
        y_true.extend(targets.numpy().tolist())
        y_pred.extend(preds.tolist())

    # Overall accuracy
    acc = accuracy_score(y_true, y_pred)
    print(f"[evaluate] Dummy accuracy: {acc:.3f}")

    # Class-wise metrics
    classes = ["Nose Breathing", "Mouth Breathing"]  # adjust if needed
    report = classification_report(y_true, y_pred, target_names=classes, digits=3)
    print("\n[evaluate] Classification Report:\n")
    print(report)

    # Percentage breakdown of predictions
    total_preds = len(y_pred)
    if total_preds > 0:
        nose_pct = (y_pred.count(0) / total_preds) * 100
        mouth_pct = (y_pred.count(1) / total_preds) * 100
        print(f"[evaluate] Predictions Breakdown: Nose {nose_pct:.1f}% | Mouth {mouth_pct:.1f}%")

    # Confusion matrix heatmap
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes)

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()

    out_path = os.path.join(_project_root(), 'models', 'confusion_matrix.png')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"[evaluate] Saved prettier confusion matrix to {out_path}")


if __name__ == '__main__':
    evaluate()
