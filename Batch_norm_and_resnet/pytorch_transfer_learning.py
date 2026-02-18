import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import models
import matplotlib.pyplot as plt
import numpy as np

def run_transfer_learning_complete():
    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Data Preparation
    transform = transforms.Compose([
        transforms.Resize(224), 
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    # CIFAR-10 Class Names (for printing)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck')

    # Load Data (Small subsets for speed)
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(torch.utils.data.Subset(trainset, range(200)), batch_size=32, shuffle=True)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(torch.utils.data.Subset(testset, range(10)), batch_size=10, shuffle=False)

    # 3. Setup Model
    print("Loading ResNet18...")
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    # Freeze Base Layers
    for param in model.parameters():
        param.requires_grad = False

    # Replace Head
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 10) 
    model = model.to(device)

    # 4. Training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)

    print("Training...")
    model.train()
    for epoch in range(20): # 1 Epoch for demo
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()
    print("Training Finished.")

    # ==========================================
    # 5. VISUALIZATION & PRINTING RESULTS
    # ==========================================
    print("\n--- Model Predictions ---")
    model.eval()
    
    # Get a single batch of test images
    dataiter = iter(testloader)
    images, labels = next(dataiter)
    images, labels = images.to(device), labels.to(device)

    # Run inference
    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

    # Print Results to Console
    print(f"{'Index':<5} | {'Actual Label':<15} | {'Predicted Label':<15} | {'Result'}")
    print("-" * 55)
    
    for i in range(len(labels)):
        actual = classes[labels[i]]
        pred = classes[predicted[i]]
        result = "✅" if actual == pred else "❌"
        print(f"{i:<5} | {actual:<15} | {pred:<15} | {result}")

# Run
if __name__ == "__main__":
    run_transfer_learning_complete()