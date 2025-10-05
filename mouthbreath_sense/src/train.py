import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from src.dataset import load_dataset
from src.utils import save_model, plot_training_curve


def _project_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def _load_config():
    cfg_path = os.path.join(_project_root(), 'config.yaml')
    print(f"[train] Loading config from {cfg_path}")
    with open(cfg_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 7 * 7, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def train():
    cfg = _load_config()
    batch_size = int(cfg.get('batch_size', 16))
    lr = float(cfg.get('learning_rate', 1e-3))
    epochs = int(cfg.get('epochs', 3))
    model_path = os.path.join(_project_root(), cfg.get('model_path', 'models/model.onnx'))

    print("[train] Preparing data loaders...")
    train_loader, _ = load_dataset(batch_size=batch_size)

    # Device selection
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == "cuda":
        print(f"[train] Using GPU: {torch.cuda.get_device_name(0)} (CUDA {torch.version.cuda})")
    else:
        print("[train] Using CPU")

    model = SimpleCNN(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    loss_history = []
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        loss_history.append(epoch_loss)
        print(f"[train] Epoch {epoch}/{epochs} - loss: {epoch_loss:.4f}")

    # Export to ONNX
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    dummy = torch.randn(1, 1, 28, 28, device=device)
    print(f"[train] Exporting model to ONNX at {model_path}")
    torch.onnx.export(
        model,
        dummy,
        model_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
    )
    print("[train] Training complete and model exported.")


if __name__ == '__main__':
    train()
