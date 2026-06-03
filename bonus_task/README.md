# Fashion-MNIST Denoising Autoencoder
 
A fully connected (MLP) denoising autoencoder for reconstructing Fashion-MNIST images, built with PyTorch.

## Architecture
 
```
ENCODER
    784 → 512 (BN + ReLU)
    512 → 256 (BN + ReLU)
    256 → 128 (BN + ReLU)
    128 → 64  (Bottleneck)
 
DECODER
     64 → 128 (BN + ReLU)
    128 → 256 (BN + ReLU)
    256 → 512 (BN + ReLU)
    512 → 784 (Sigmoid)
```

**Features**
- 4 layers in both encoder and decoder for good extraction of features
- Bottleneck of 64 dimensions compresses 784-dimensional input into its compact features
- Sigmoid output keeps pixel values in [0, 1]
- BCE loss for sharper reconstructions over MSE
- Gaussian noise added to inputs during training for denoising

## Hyperparameters
 
| Parameter | Value |
|---|---|
| Learning Rate | 1e-4 |
| Weight Decay | 1e-4 |
| Batch Size | 64 |
| Optimiser | Adam |
| Loss Function | BCE |
| Noise Scale | 0.05 |
| Bottleneck Size | 64 |
| Epochs | 30 |

## Findings
- Augmentation does not work here as instead of classification we wish to reconstruct the pixels, and any change to their orientation or look will lead to faulty learning and distorted outputs.
- Dropout regularisation was not used in this case as the autoencoder is already extraction compact and consise features in the bottlenack, so deactivating random neurons would make the training process more difficult.
- Scheduling is also not used here as there is not much skew in the loss values as we go through the epochs.
- Instead of dropout, denoising has been used with a factor of 0.05 to inject noise into the images during training followed by comparison with the original images so that the model learns the actual features instead of memorising the pixel values.
- The denoising factor was experimented with, and a factor of 0.1-0.2 was found to be very high for such a simple model, while 0.05-0.075 seemed to be a sweet spot, with not much difference at all between the two
- The bottleneck size was also varied to be 16, 24, 32, 64. 64 came to give the best results, showing that upto a certain range, increasing the number of neurons in the bottleneck gives better extraction of features.
- The number of hidden layers was kept at 4 to maximise the reconstruction quality.
- BCE was chosen over MSE as the former treats the pixel values from 0 to 1 as a probability distribution over actual values, which gives better results.

## Outputs
 
| File | Description |
|---|---|
| `reconstructions.png` | Side-by-side comparison of original vs reconstructed images from the final epoch |
 
## Limitations
Being a fully connected (MLP) autoencoder, spatial relationships between pixels are lost during flattening. This results in blurry reconstructions, using a convolutional autoencoder would produce significantly sharper results.
