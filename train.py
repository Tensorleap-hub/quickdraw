import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from tqdm import tqdm
from dataset import QuickDrawDataset
from config import NPY_PATH, NDJSON_PATH


# Configuration
BATCH_SIZE = 128
EPOCHS = 5
LEARNING_RATE = 0.001
MAX_SAMPLES_PER_CLASS = None  # Set to None for full dataset
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
MODEL_SAVE_PATH = "resnet18_quickdraw.pth"
ONNX_SAVE_PATH = "resnet18_quickdraw.onnx"


def create_resnet18_for_quickdraw(num_classes: int) -> nn.Module:
    """
    Create a ResNet18 model adapted for QuickDraw dataset.
    - Modified first conv layer: 1 input channel (grayscale), smaller kernel for 28x28 images
    - Modified final FC layer: output num_classes
    """
    model = resnet18(weights=None)
    
    # Modify first conv layer for single channel input and smaller images
    # Original: Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
    # Modified: Conv2d(1, 64, kernel_size=3, stride=1, padding=1) for 28x28 images
    model.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
    
    # Remove max pooling to preserve spatial dimensions for small images
    model.maxpool = nn.Identity()
    
    # Modify final fully connected layer for our number of classes
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    return model


def train_one_epoch(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, 
                    optimizer: optim.Optimizer, device: torch.device) -> float:
    """Train for one epoch and return average loss."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    return running_loss / len(dataloader)


def validate(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, 
             device: torch.device) -> tuple:
    """Validate and return average loss and accuracy."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    accuracy = 100. * correct / total
    avg_loss = running_loss / len(dataloader)
    return avg_loss, accuracy


def export_to_onnx(model: nn.Module, save_path: str, device: torch.device):
    """Export the trained model to ONNX format."""
    model.eval()
    
    # Create dummy input with correct shape (batch, channels, height, width)
    dummy_input = torch.randn(1, 1, 28, 28).to(device)
    
    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        save_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    print(f"Model exported to ONNX: {save_path}")


def main():
    print(f"Using device: {DEVICE}")
    
    # Load dataset
    print("Loading dataset...")
    dataset = QuickDrawDataset(NPY_PATH, NDJSON_PATH, max_samples_per_class=MAX_SAMPLES_PER_CLASS)
    
    # Split into train and validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    print(f"Train samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    
    # Create model
    num_classes = len(dataset.classes)
    model = create_resnet18_for_quickdraw(num_classes).to(DEVICE)
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    # Training loop
    best_acc = 0.0
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print("-" * 40)
        
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)
        scheduler.step()
        
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, MODEL_SAVE_PATH)
            print(f"Saved best model with accuracy: {val_acc:.2f}%")
    
    # Load best model and export to ONNX
    print("\nLoading best model for ONNX export...")
    checkpoint = torch.load(MODEL_SAVE_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from epoch {checkpoint['epoch']+1} with accuracy {checkpoint['val_acc']:.2f}%")
    
    export_to_onnx(model, ONNX_SAVE_PATH, DEVICE)
    
    print("\nTraining complete!")
    print(f"Best validation accuracy: {best_acc:.2f}%")
    print(f"PyTorch model saved to: {MODEL_SAVE_PATH}")
    print(f"ONNX model saved to: {ONNX_SAVE_PATH}")


if __name__ == "__main__":
    main()
