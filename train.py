import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from constants import IMAGES_DIR, LIST_FILE, TEST_FILE, MODEL_PATH

class OxfordPetsDataset(Dataset):
    def __init__(self, list_file, images_dir, transform=None):
        self.images_dir = images_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        # Parse list file
        with open(list_file, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue
            
            parts = line.split()
            
            # Validate line format
            if len(parts) != 4:
                continue
            
            image_name, class_id, species, breed_id = parts
            
            # Construct image path and label
            image_path = os.path.join(images_dir, image_name + '.jpg')
            
            try:
                label = int(class_id) - 1  # Convert class ID to 0-based index
                if label < 0 or label >= 37:
                    raise ValueError(f"Invalid CLASS-ID: {class_id}")
            except ValueError as e:
                print(f"Skipping line due to error: {e}")
                continue
            
            self.image_paths.append(image_path)
            self.labels.append(label)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load and preprocess image
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a placeholder image in case of error
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# Simple CNN model definition
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=37):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 28 * 28)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

def get_class_names(list_file):
    class_id_to_name = {}
    
    with open(list_file, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        line = line.strip()
        
        # Skip empty lines and comments
        if not line or line.startswith('#'):
            continue
        
        parts = line.split()
        
        if len(parts) != 4:
            continue
        
        image_name, class_id, species, breed_id = parts
        
        # Extract breed name
        breed_name = image_name.split('_')[0]
        
        try:
            class_id_int = int(class_id)
            if class_id_int < 1 or class_id_int > 37:
                raise ValueError(f"Invalid CLASS-ID: {class_id}")
        except ValueError as e:
            print(f"Skipping line due to error: {e}")
            continue
        
        # Map class ID to breed name
        if class_id_int not in class_id_to_name:
            class_id_to_name[class_id_int] = breed_name
        else:
            if class_id_to_name[class_id_int] != breed_name:
                print(f"Warning: CLASS-ID {class_id_int} is associated with multiple breed names.")
    
    return [class_id_to_name.get(i, f"Unknown_{i}") for i in range(1, 38)]

def train_and_evaluate(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=10):
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Training'):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Validation'):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_epoch_loss = val_loss / len(val_loader.dataset)
        val_epoch_acc = val_correct / val_total
        
        print(f'Epoch {epoch+1}/{num_epochs} - Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f} - '
              f'Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_acc:.4f}')
        
        # Save best model
        if val_epoch_acc > best_val_acc:
            best_val_acc = val_epoch_acc
            torch.save(model.state_dict(), MODEL_PATH)
            print('Best model saved')

    print(f'Best Validation Accuracy: {best_val_acc:.4f}')

def evaluate_model(model, test_loader, criterion, device, class_names):
    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc='Testing'):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    final_test_loss = test_loss / len(test_loader.dataset)
    final_test_acc = test_correct / test_total
    
    print(f'Test Loss: {final_test_loss:.4f}, Test Accuracy: {final_test_acc:.4f}')
    
    # Display confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()
    
    # Display classification report
    print('Classification Report:')
    print(classification_report(all_labels, all_preds, target_names=class_names))

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f'Using device: {device}')
    
    # Get class names
    class_names = get_class_names(LIST_FILE)
    print(f'Class names ({len(class_names)}): {class_names}')
    
    # Data preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create dataset
    dataset = OxfordPetsDataset(list_file=LIST_FILE, images_dir=IMAGES_DIR, transform=transform)
    print(f'Total samples in dataset: {len(dataset)}')
    
    # Split dataset into train and validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    print(f'Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}')
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # Initialize model
    model = SimpleCNN(num_classes=37).to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train and evaluate the model
    num_epochs = 10
    train_and_evaluate(model, train_loader, val_loader, criterion, optimizer, device, num_epochs)
    
    # Create test dataset
    test_dataset = OxfordPetsDataset(list_file=TEST_FILE, images_dir=IMAGES_DIR, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    print(f'Test samples: {len(test_dataset)}')
    
    # Evaluate on test data
    evaluate_model(model, test_loader, criterion, device, class_names)

if __name__ == '__main__':
    main()
