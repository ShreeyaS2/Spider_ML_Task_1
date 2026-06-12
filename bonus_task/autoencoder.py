#Code
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

#For fixed initialisation
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

transform = transforms.Compose([
    transforms.ToTensor(),
])

train_set = torchvision.datasets.FashionMNIST(root='./data', download=True, train=True, transform=transform)
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)

test_set = torchvision.datasets.FashionMNIST(root='./data', download=True, train=False, transform=transform)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

DEVICE= torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Autoencoder
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 784),
            nn.Sigmoid() 
        )
    
    def forward(self, x):
        encode=self.encoder(x)
        decode=self.decoder(encode)
        return decode
    
#Training
def train(model, loader, criterion, optimizer, device):
    model.train()

    for images, _ in loader: #every 64 images and labels in the training set
        images = images.to(device)
        noise = torch.randn_like(images) * 0.05
        noisy_images = images + noise
        noisy_images = torch.clamp(noisy_images, 0.0, 1.0)
        
        optimizer.zero_grad()
        logits = model.forward(noisy_images)
        loss   = criterion(logits, images.reshape(-1, 784)) #compare output to original image (flattened)
        loss.backward()
        optimizer.step()

#Evaluate(no gradient calculation)
@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    for images, _ in loader:
        images = images.to(device)

        logits = model.forward(images)
        loss   = criterion(logits, images.reshape(-1, 784))
        total_loss += loss.item()*images.size(0)
    avg_loss = total_loss / len(loader.dataset)
    return images, logits, avg_loss        
        
model=Autoencoder().to(DEVICE)
criterion=nn.BCELoss()

optimizer=optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

output=[]
for epoch in range(30):
    train(model, train_loader, criterion, optimizer, DEVICE)
    images, logits, loss = evaluate(model, test_loader, criterion, DEVICE)       
    print(f"Epoch {epoch+1}, Loss: {loss:.4f}")
    output.append((images, logits)) #last batch value for every epoch

# grab the last epoch's images and reconstructions
images, reconstructed = output[-1]

images = images.detach().cpu()
reconstructed = reconstructed.detach().cpu()

#reconstructed is (B, 784) — needs reshaping
reconstructed = reconstructed.reshape(-1, 1, 28, 28)

fig, axes = plt.subplots(2, 10, figsize=(15, 3))
fig.suptitle("Original vs Reconstructed", fontsize=13, fontweight="bold")

for i in range(10):
    # top row — original images
    axes[0, i].imshow(images[i].squeeze(), cmap="gray")
    axes[0, i].axis("off")
    if i == 0:
        axes[0, i].set_title("Original", loc="left")
    
    # bottom row — reconstructed images
    axes[1, i].imshow(reconstructed[i].squeeze(), cmap="gray")
    axes[1, i].axis("off")
    if i == 0:
        axes[1, i].set_title("Reconstructed", loc="left")

plt.tight_layout()
plt.savefig("reconstructions.png", dpi=150, bbox_inches="tight")
plt.show()

