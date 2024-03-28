import torch
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch import nn, optim

# Select the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the dataset
def get_dataloader(data_path, batch_size=32):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = datasets.ImageFolder(root=data_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    return dataloader

# Initialize models
def initialize_model(model_name):
    if model_name == "resnet50":
        model = models.resnet50(pretrained=True)
    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(pretrained=True)
    elif model_name == "vgg16":
        model = models.vgg16(pretrained=True)
    else:
        raise ValueError("Unknown model name")

    model = model.to(device)
    return model

# Training loop
def train_model(model, dataloader, epochs=1):
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

# Main function to run the experiment
def run_experiment(model_name, data_path):
    dataloader = get_dataloader(data_path)
    model = initialize_model(model_name)
    train_model(model, dataloader)
    
model_names = ["resnet50"]
# model_names = ["resnet50", "efficientnet_b0", "vgg16"]
data_path = "./imagenette2-160"

for model_name in model_names:
    print(f"Training {model_name}...")
    run_experiment(model_name, data_path)
