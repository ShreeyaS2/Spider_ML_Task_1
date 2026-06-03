# Fashion-MNIST Classifier
 
A custom multi-branch neural network for classifying Fashion-MNIST images, built with PyTorch.

## Architecture
- Input: 28×28 grayscale images (flattened to 784)
- Shared stem: 784 → 16 (BatchNorm + ReLU + Dropout)
- Branch A: 16 → 8 → 8 with residual skip connection
- Branch B: 16 → 12 → 8 (BatchNorm + ReLU + Dropout)
- Output: Concatenated branches (16) → 10 classes

**Features**
- He (Kaiming) initialisation on all linear layers
- Batch Normalisation after every hidden layer
- Residual skip connection on Branch A
- Dropout (0.2) for regularisation
- L2 regularisation via Adam `weight_decay=1e-4`
- Learning rate scheduler: `ReduceLROnPlateau(patience=3, factor=0.5)`
- Seed of 42 for consistent weight initialisation

## Results
 
| Metric | Value |
|---|---|
| Best Validation Accuracy | 85.73% |
| Generalisation Gap | ~0% |

<img width="766" height="123" alt="Screenshot 2026-06-02 214435" src="https://github.com/user-attachments/assets/5dc6b54b-bd24-4aa4-90b8-a0c2f751d478" />

## Hyperparameters
 
| Parameter | Value |
|---|---|
| Learning Rate | 1e-3 |
| Weight Decay | 1e-4 |
| Batch Size | 64 |
| Optimiser | Adam |
| Loss Function | Cross Entropy |
| Dropout | 0.15 |
| Epochs | 50 |
| Seed | 42 |

## Findings
- Augmentation gave worse results for the model as a simple model with ~10,000 parameters aand less nodes in the hidden layers is not able to properly learn further skewed inputs caused by augmentation.
- The dropout led to a successful lowering of the generalisation gap as it gets turned off during evaluation causing the evaluation to take place at full computational power.
- After about 40-45 epochs, the graphs become very consistent and don't skew with decent training and validation results, with the validation loss and accuracy being better.

## Outputs
 
| File | Description |
|---|---|
| `model_weights.pkl` | Best model weights |
| `training_curves.png` | Loss and accuracy plots over 50 epochs |
| `submission.csv` | Predictions on the test set with class labels |
