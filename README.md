# MNIST — FCNN vs CNN

MNIST digit classification comparing a fully connected neural network (FCNN) against a convolutional neural network (CNN). The CNN uses batch normalization and data augmentation to improve generalization and training stability.

## Dataset
- MNIST (28×28 grayscale digits, 0–9)
- Train/Val/Test split: 50,000 / 10,000 / 10,000
- Normalization: mean=0.1307, std=0.3081
- Data augmentation (CNN): random rotations (±10°), shifts (±5%), scaling (0.95–1.05) 

## Models
### FCNN
- 784 → 256 (ReLU) → 10 :contentReference[oaicite:2]{index=2}
- Optimizer: Adam (lr=0.001), batch size 64, epochs 10 :contentReference[oaicite:3]{index=3}

### CNN
- Conv(1→32, 3×3) + BatchNorm + ReLU + MaxPool
- Conv(32→64, 3×3) + BatchNorm + ReLU + MaxPool
- FC(64×7×7→128) + ReLU → FC(128→10) 
- Optimizer: Adam (lr=0.001), batch size 64, epochs 15 :contentReference[oaicite:5]{index=5}

## Results
| Metric | FCNN | CNN |
|---|---:|---:|
| Parameters | 203,530 | 100,416 |
| Validation Accuracy | 97.5% | 98.8% |
| Test Accuracy | 97.4% | 98.9% |

CNN improved test accuracy by ~1.5% while using fewer parameters. 

## Notes
- FCNN learned fast but plateaued early and showed mild overfitting.
- CNN training stayed stable and generalized better; batch normalization and augmentation helped. 

## How to Run
1. Install:
   ```bash
   pip install -r requirements.txt
2. Run: 