import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm
import os

# Parameters
num_epochs = 100
batch_size = 512
learning_rate = 0.001
trigger_size = 3
trigger_value = 255
trigger_label_target = 1
poisoned_fraction = 0.1
num_models_per_class = 1
num_classes = 10
sigma = 0.1
learning_rate_b3d = 0.05
k = 50
T = 100
minibatch_size = 128

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822 ,0.4465), (0.2470, 0.2434, 0.2616))
])

# CIFAR-10 dataset
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# DataLoader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Define ResNet-18 model
def create_model():
    model = torchvision.models.resnet18(pretrained=False, num_classes=10).to(device)
    return model

# Loss and optimizer
def get_optimizer(model):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    return optimizer

criterion = nn.CrossEntropyLoss()

# Function to add trigger
def add_trigger(images, trigger_value, trigger_size):
    images[:, -trigger_size:, -trigger_size:] = trigger_value
    return images

# Function to poison dataset
def poison_dataset(images, labels, trigger_value, trigger_size, trigger_label_target, poisoned_fraction):
    poisoned_data = images.copy()
    poisoned_labels = labels.copy()
    for class_label in range(10):  # Process each class separately
        class_indices = np.where(labels == class_label)[0]
        num_poisoned = int(len(class_indices) * poisoned_fraction)
        poisoned_indices = class_indices[:num_poisoned]  # Fixed: first 20%

        for idx in poisoned_indices:
            poisoned_data[idx] = add_trigger(poisoned_data[idx], trigger_value, trigger_size)
            poisoned_labels[idx] = trigger_label_target

    return poisoned_data, poisoned_labels

# Custom CIFAR-10 dataset class for poisoned data
class PoisonedCIFAR10(torchvision.datasets.CIFAR10):
    def __init__(self, poisoned_data, poisoned_targets, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data = poisoned_data
        self.targets = poisoned_targets

# Train the model
def train(model, train_loader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in tqdm(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

# Test the model
def test(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Test Accuracy: {100 * correct / total:.2f}%')

# Function to test model on triggered test set
def test_with_trigger(model, test_loader, trigger_value, trigger_size):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.clone().to(device)
            images = add_trigger(images, trigger_value, trigger_size)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Test Accuracy with Trigger: {100 * correct / total:.2f}%')

# B3D detection function
def b3d_detection(model, clean_images, num_classes, sigma=0.1, learning_rate=0.05, k=50, T=100, minibatch_size=128):
    d = clean_images.shape[1:]  # dimension of input images

    def loss_fn(m, p, c, model, clean_images):
        loss = 0
        for xi in clean_images:
            xi = torch.tensor(xi, dtype=torch.float32).to(device)
            x_triggered = (1 - m) * xi + m * p
            x_triggered = x_triggered.unsqueeze(0)
            y_pred = model(x_triggered)
            y_true = torch.tensor([c]).to(device)
            loss += nn.CrossEntropyLoss()(y_pred, y_true)
        return loss

    def optimize_trigger(c, model, clean_images, sigma, learning_rate, k, T):
        θm = torch.rand(*d, device=device, requires_grad=True)
        θp = torch.rand(*d, device=device, requires_grad=True)
        optimizer_m = optim.Adam([θm], lr=learning_rate)
        optimizer_p = optim.Adam([θp], lr=learning_rate)
        
        for t in range(T):
            g_m = torch.zeros_like(θm)
            g_p = torch.zeros_like(θp)
            for _ in range(k):
                # Randomly select a minibatch from clean_images
                minibatch_indices = np.random.choice(len(clean_images), minibatch_size, replace=False)
                minibatch = clean_images[minibatch_indices]

                m = torch.bernoulli(torch.full(d, 0.5)).to(device)
                p = θp + sigma * torch.randn_like(θp).to(device)
                loss = loss_fn(m, p, c, model, minibatch)
                loss.backward()
                with torch.no_grad():
                    g_m += 2 * (m - torch.sigmoid(θm)) * loss.item()
                    g_p += (p - θp) / sigma * loss.item()
            
            # Perform Adam update with gradients scaled by 1/k
            optimizer_m.zero_grad()
            optimizer_p.zero_grad()
            θm.grad = g_m / k
            θp.grad = g_p / (k * sigma)
            optimizer_m.step()
            optimizer_p.step()
        
        m_opt = (θm > 0.5).float()
        p_opt = θp
        return m_opt.cpu().detach().numpy(), p_opt.cpu().detach().numpy()

    triggers = {}
    for c in range(num_classes):
        m_opt, p_opt = optimize_trigger(c, model, clean_images, sigma, learning_rate, k, T)
        triggers[c] = (m_opt, p_opt)
    
    l1_norms = {c: np.sum(np.abs(m)) for c, (m, p) in triggers.items()}
    median_l1 = np.median(list(l1_norms.values()))
    outliers = [c for c, l1 in l1_norms.items() if l1 < median_l1 / 4]
    
    return outliers, l1_norms, median_l1

# Directory to save models
os.makedirs('models', exist_ok=True)

# Train and save backdoor models
seed = 0
for class_label in range(num_classes):
    for i in range(num_models_per_class):
        if os.path.exists(f'models/backdoored_seed={seed}_class_{class_label}_model_{i}.pth'):
            print(f'Backdoor Class {class_label} Model {i} already exists. Skipping training.')
            continue
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        print(f'Training backdoor model for class {class_label}, model {i}')
        
        # Create and poison dataset
        train_data = train_dataset.data
        train_targets = np.array(train_dataset.targets)
        poisoned_data, poisoned_targets = poison_dataset(train_data, train_targets, trigger_value, trigger_size, trigger_label_target, poisoned_fraction)

        # Create poisoned dataset
        poisoned_train_dataset = PoisonedCIFAR10(poisoned_data, poisoned_targets, root='./data', train=True, download=True, transform=transform)
        poisoned_train_loader = torch.utils.data.DataLoader(dataset=poisoned_train_dataset, batch_size=batch_size, shuffle=True)

        # Create model and optimizer
        model = create_model()
        optimizer = get_optimizer(model)

        # Train model
        train(model, poisoned_train_loader, criterion, optimizer, num_epochs)

        # Save model
        torch.save(model.state_dict(), f'models/backdoored_seed={seed}_class_{class_label}_model_{i}.pth')
        seed += 1

# Train and save normal models
for i in range(num_classes * num_models_per_class):
    seed = 100 + i
    if os.path.exists(f'models/normal_seed={seed}_{i}.pth'):
        print(f'Normal Model {i} already exists.Skipping training.')
        continue
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    print(f'Training normal model {i}')
    
    # Create model and optimizer
    model = create_model()
    optimizer = get_optimizer(model)

    # Train model
    train(model, train_loader, criterion, optimizer, num_epochs)

    # Save model
    torch.save(model.state_dict(), f'models/normal_seed={seed}_{i}.pth')

# B3D detection on all models
clean_images, _ = next(iter(test_loader))  # Get a batch of clean images

# Detect backdoored models
for model_type in ['backdoored', 'normal']:
    for i in range(num_classes * num_models_per_class):
        if model_type == 'backdoored':
            model_path = f'models/backdoored_seed={i}_class_{i%num_models_per_class}_model_{i%num_models_per_class}.pth'
        else:
            model_path = f'models/normal_seed={100 + i}_{i}.pth'
        
        model = create_model()
        model.load_state_dict(torch.load(model_path))
        model.eval()

        print(f'Detecting {model_type} model {i}')
        outliers, l1_norms, median_l1 = b3d_detection(model, clean_images.numpy(), num_classes, sigma, learning_rate_b3d, k, T)

        if outliers:
            print(f'Model {model_type}_{i} detected as backdoored. Outliers: {outliers}')
        else:
            print(f'Model {model_type}_{i} detected as clean.')

        print(f'L1 norms: {l1_norms}')
        print(f'Median L1 norm: {median_l1}')
        print(f'Outliers: {outliers}')
