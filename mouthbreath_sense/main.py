import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from src.train import train
from src.evaluate import evaluate
from src.inference import predict
import numpy as np

def main():
    print("\nMouthBreathSense CLI")
    print("1: Train model")
    print("2: Evaluate model")
    print("3: Run inference (dummy)")
    choice = input("Select an option (1/2/3): ").strip()

    if choice == '1':
        train()
    elif choice == '2':
        evaluate()
    elif choice == '3':
        print("[cli] Running inference with a random dummy input...")
        sample = np.random.randn(28, 28).astype('float32')
        pred = predict(sample)
        print(f"[cli] Prediction: {pred}")
    else:
        print("[cli] Invalid choice.")


if __name__ == '__main__':
    main()
