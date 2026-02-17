import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import models

def run_transfer_learning():
    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Data Preparation (CIFAR-10)
    # We use standard ImageNet stats for normalization because ResNet expects them
    transform = transforms.Compose([
        transforms.Resize(224), # ResNet expects 224x224
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    # Load CIFAR-10
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    
    # Create a small subset for this exercise (e.g., 500 images) to speed up testing
    subset_indices = range(500) 
    train_subset = torch.utils.data.Subset(trainset, subset_indices)
    
    trainloader = torch.utils.data.DataLoader(train_subset, batch_size=32,
                                              shuffle=True, num_workers=2)

    # 3. Load Pre-trained ResNet18
    print("Loading ResNet18...")
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    # 4. Freeze all layers
    for param in model.parameters():
        param.requires_grad = False

    # 5. Replace the Final Layer
    # ResNet's final layer is called 'fc'. We replace it with a new Linear layer.
    # New layers have requires_grad=True by default.
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 10) # 10 output classes for CIFAR-10

    model = model.to(device)

    # 6. Define Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    
    # Only optimize the parameters of the final layer!
    optimizer = optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)

    # 7. Fine-tuning Loop
    print("Starting Fine-tuning...")
    model.train()
    
    for epoch in range(1): # Run 1 epoch for demonstration
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + Backward + Optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 5 == 0:
                print(f'[Epoch {epoch + 1}, Batch {i + 1}] loss: {running_loss / 5:.3f}')
                running_loss = 0.0

    print('Finished Fine-tuning')

# Run the function
if __name__ == "__main__":
    run_transfer_learning()