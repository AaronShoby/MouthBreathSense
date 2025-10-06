import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt
from src.dataset import prepare_datasets, FeatureConfig, LoaderConfig

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------
# CNN-GRU model (same as in train.py)
# -------------------------------
class CNN_GRU(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(CNN_GRU, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        self.gru = nn.GRU(input_size=128, hidden_size=64, num_layers=1,
                          batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(64 * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        out, _ = self.gru(x)
        out = out.mean(dim=1)
        return self.fc(out)

# -------------------------------
# Evaluation function
# -------------------------------
def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss, total, correct = 0.0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            x, y = batch["x"].to(DEVICE), batch["y"].to(DEVICE)
            outputs = model(x)
            loss = criterion(outputs, y)

            total_loss += loss.item() * x.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy, np.array(all_labels), np.array(all_preds)

# -------------------------------
# Main evaluation script
# -------------------------------
def main():
    print("[evaluate] Starting evaluation...")

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    model_path = os.path.join(repo_root, "mouthbreath_sense", "models", "best_model.pt")
    feat_cfg, load_cfg = FeatureConfig(), LoaderConfig(batch_size=16)

    print("[evaluate] Loading dataset...")
    _, _, test_ds, meta = prepare_datasets(repo_root, feat_cfg, load_cfg, include_icbhi=False)
    test_dl = DataLoader(test_ds, batch_size=load_cfg.batch_size, shuffle=False)

    in_channels = meta["feature_shape"][0]
    num_classes = len(meta["classes"])

    print("[evaluate] Loading model from:", model_path)
    model = CNN_GRU(in_channels, num_classes).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    criterion = nn.CrossEntropyLoss()

    print(f"[evaluate] Using device: {DEVICE.upper()}")
    loss, acc, labels, preds = evaluate(model, test_dl, criterion)

    print("\n✅ Evaluation Complete!")
    print(f"Test Loss: {loss:.4f} | Test Accuracy: {acc:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(labels, preds)
    print("\nConfusion Matrix:\n", cm)

    plt.figure(figsize=(5, 5))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix (Evaluation)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar()
    save_path = os.path.join(repo_root, "mouthbreath_sense", "models", "confusion_matrix_eval.png")
    plt.savefig(save_path)
    plt.close()
    print(f"[evaluate] Confusion matrix saved to: {save_path}")

    # Classification Report
    print("\nClassification Report:")
    print(classification_report(labels, preds, target_names=meta["classes"]))

if __name__ == "__main__":
    main()
