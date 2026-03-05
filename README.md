# MNIST — FCNN vs CNN

MNIST digit classification comparing a fully connected neural network (FCNN) against a convolutional neural network (CNN). The CNN uses batch normalization and data augmentation to improve generalization and training stability.

## Dataset
- MNIST (28×28 grayscale digits)
- Train/Val/Test: 50,000 / 10,000 / 10,000
- Normalization: mean=0.1307, std=0.3081
- Augmentation (CNN): rotations (±10°), shifts (±5%), scaling (0.95–1.05)

## Models
### FCNN
784 → 256 (ReLU) → 10  
Optimizer: Adam (lr=0.001), batch=64, epochs=10  
Params: 203,530

### CNN
Conv(1→32, 3×3, pad=1) + BN + ReLU + MaxPool  
Conv(32→64, 3×3, pad=1) + BN + ReLU + MaxPool  
FC(64×7×7→128) + ReLU → FC(128→10)  
Optimizer: Adam (lr=0.001), batch=64, epochs=15  
Augmentation enabled  
Params: 100,416

## Results
| Metric | FCNN | CNN |
|---|---:|---:|
| Train Acc | 98.2% | 99.1% |
| Val Acc | 97.5% | 98.8% |
| Test Acc | 97.4% | 98.9% |
| Params | 203,530 | 100,416 |

## How to Run
1. Install:
   ```bash
   pip install -r requirements.txt
2. Train/evaluate both models:
   ```bash
   python train.py
