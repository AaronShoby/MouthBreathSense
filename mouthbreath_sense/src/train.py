import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import numpy as np
import matplotlib.pyplot as plt
from src.dataset import prepare_datasets, FeatureConfig, LoaderConfig

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------
# CNN-GRU Model
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

        self.gru = nn.GRU(input_size=128, hidden_size=64, num_layers=1, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(64 * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = x.permute(0, 2, 1)  # [B, T, C]
        out, _ = self.gru(x)
        out = out.mean(dim=1)
        return self.fc(out)

# -------------------------------
# Training utilities
# -------------------------------
def train_one_epoch(model, dataloader, optimizer, criterion):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for batch in dataloader:
        x, y = batch["x"].to(DEVICE), batch["y"].to(DEVICE)
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x.size(0)
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    return running_loss / total, correct / total


def evaluate(model, dataloader, criterion):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            x, y = batch["x"].to(DEVICE), batch["y"].to(DEVICE)
            outputs = model(x)
            loss = criterion(outputs, y)
            running_loss += loss.item() * x.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    acc = correct / total
    return running_loss / total, acc, np.array(all_labels), np.array(all_preds)


# -------------------------------
# Main training pipeline
# -------------------------------
def main():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    feat_cfg = FeatureConfig(sample_rate=16000, duration=5.0, n_mels=64, n_mfcc=40)
    load_cfg = LoaderConfig(batch_size=16, num_workers=4, val_split=0.1, test_split=0.1)

    print("[train] Preparing datasets...")
    train_ds, val_ds, test_ds, meta = prepare_datasets(repo_root, feat_cfg, load_cfg, include_icbhi=False)
    print("[train] Dataset ready:", meta)

    in_channels = meta["feature_shape"][0]
    num_classes = len(meta["classes"])

    train_dl = DataLoader(train_ds, batch_size=load_cfg.batch_size, shuffle=True, num_workers=load_cfg.num_workers)
    val_dl = DataLoader(val_ds, batch_size=load_cfg.batch_size, shuffle=False, num_workers=load_cfg.num_workers)
    test_dl = DataLoader(test_ds, batch_size=load_cfg.batch_size, shuffle=False, num_workers=load_cfg.num_workers)

    model = CNN_GRU(in_channels=in_channels, num_classes=num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    epochs = 15

    best_val_acc = 0
    model_dir = os.path.join(repo_root, "mouthbreath_sense", "models")
    os.makedirs(model_dir, exist_ok=True)

    print(f"[train] Using device: {DEVICE.upper()} ({torch.cuda.get_device_name(0) if DEVICE == 'cuda' else 'CPU'})")

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_dl, optimizer, criterion)
        val_loss, val_acc, _, _ = evaluate(model, val_dl, criterion)
        print(f"Epoch [{epoch}/{epochs}] | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(model_dir, "best_model.pt"))

    print("\n✅ Training complete. Best Val Acc:", best_val_acc)

    # Test evaluation
    print("\n🔍 Evaluating on test set...")
    model.load_state_dict(torch.load(os.path.join(model_dir, "best_model.pt")))
    test_loss, test_acc, labels, preds = evaluate(model, test_dl, criterion)
    print(f"Test Accuracy: {test_acc:.4f}")

    # Confusion matrix
    cm = confusion_matrix(labels, preds)
    print("\nConfusion Matrix:\n", cm)
    plt.imshow(cm, cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(os.path.join(model_dir, "confusion_matrix.png"))
    plt.close()

    # Classification report
    print("\nClassification Report:")
    print(classification_report(labels, preds, target_names=meta["classes"]))

    # Export to ONNX
    dummy_input = torch.randn(1, in_channels, 313).to(DEVICE)
    onnx_path = os.path.join(model_dir, "model.onnx")
    torch.onnx.export(
        model, dummy_input, onnx_path,
        input_names=["input"], output_names=["output"],
        opset_version=13
    )
    print(f"\n🧠 ONNX model exported successfully → {onnx_path}")

if __name__ == "__main__":
    main()
