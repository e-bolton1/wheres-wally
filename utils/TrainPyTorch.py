import os
import time
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import gc


def train_waldo_pytorch(data_path='Data', epochs=15, test_size=0.10, save_path=None, img_size=64, batch_size=32, learning_rate=0.001):
    """
    Train a PyTorch model to classify Waldo vs NotWaldo images.
    Follows the Train.py structure with PyTorch implementation.
    
    Args:
        data_path: Path to Data directory containing Waldo/ and NotWaldo/ folders
        epochs: Number of training epochs (default 15)
        test_size: Fraction of data to use for testing (default 0.10)
        save_path: Path to save trained model weights (.pth file)
        img_size: Size to resize all images to (default 64, matching Train.py)
        batch_size: Batch size for training (default 32)
        learning_rate: Learning rate for optimizer (default 0.001)
    
    Returns:
        model: Trained PyTorch model
        X_test: Test images (numpy array)
        Y_test: Test labels (numpy array)
        metrics: Dictionary containing test_loss and test_accuracy
    """
    
    # Import Model.py
    import sys
    utils_path = os.path.dirname(os.path.abspath(__file__))
    if utils_path not in sys.path:
        sys.path.insert(0, utils_path)
    import Model
    
    # Start overall timer
    overall_start_time = time.time()
    
    # Clear memory before starting
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    waldo_dir = os.path.join(data_path, 'Waldo')
    not_waldo_dir = os.path.join(data_path, 'NotWaldo')
    
    # Load training data
    X, Y = [], []
    
    # Timer for data loading
    load_start_time = time.time()
    
    print("Loading Waldo images (label=1)...")
    for img_file in os.listdir(waldo_dir):
        if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(waldo_dir, img_file)
            try:
                img = Image.open(img_path).convert('RGB')
                img = img.resize((img_size, img_size))
                img_array = np.array(img, dtype=np.float32) / 255.0  # Normalize to [0, 1]
                X.append(img_array)
                Y.append(1)
            except Exception as e:
                print(f"Error loading {img_file}: {e}")
    
    print("Loading NotWaldo images (label=0)...")
    for img_file in os.listdir(not_waldo_dir):
        if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(not_waldo_dir, img_file)
            try:
                img = Image.open(img_path).convert('RGB')
                img = img.resize((img_size, img_size))
                img_array = np.array(img, dtype=np.float32) / 255.0  # Normalize to [0, 1]
                X.append(img_array)
                Y.append(0)
            except Exception as e:
                print(f"Error loading {img_file}: {e}")
    
    # Convert to numpy arrays
    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.float32)
    
    load_end_time = time.time()
    load_duration = load_end_time - load_start_time
    
    print(f"\nTotal images: {len(X)}")
    print(f"Waldo images: {int(sum(Y))}, NotWaldo images: {int(len(Y) - sum(Y))}")
    print(f"Data loading time: {load_duration:.2f} seconds")
    
    # Split data (same as Train.py: 10% test)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=42)
    
    print(f'\nX_train shape: {X_train.shape}')
    print(f'{X_train.shape[0]} train samples')
    print(f'{X_test.shape[0]} test samples')
    
    # Convert to PyTorch tensors (channels first: C, H, W)
    X_train_tensor = torch.from_numpy(X_train).permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
    X_test_tensor = torch.from_numpy(X_test).permute(0, 3, 1, 2)
    Y_train_tensor = torch.from_numpy(Y_train).long()
    Y_test_tensor = torch.from_numpy(Y_test).long()
    
    # Free up numpy arrays
    del X
    gc.collect()
    
    # Create DataLoaders
    train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, Y_test_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Get model from Model.py
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Model.get_conv(input_shape=(3, img_size, img_size)).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    print(f"\nTraining on {device}...")
    training_start_time = time.time()
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        
        val_loss /= len(test_loader)
        val_accuracy = correct / total
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        
        print(f'Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}')
    
    training_end_time = time.time()
    training_duration = training_end_time - training_start_time
    
    # Free training data
    del X_train_tensor, Y_train_tensor, train_dataset, train_loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Final evaluation
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            test_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
    
    test_loss /= len(test_loader)
    test_accuracy = correct / total
    
    print(f'\nTest Loss: {test_loss:.4f}')
    print(f'Test Accuracy: {test_accuracy:.4f}')
    
    # Save model weights
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        torch.save(model.state_dict(), save_path)
        print(f"\nModel saved to {save_path}")
    
    # Calculate and print timing summary
    overall_end_time = time.time()
    overall_duration = overall_end_time - overall_start_time
    avg_epoch_time = training_duration / epochs
    
    print("\n" + "="*50)
    print("TRAINING TIMING SUMMARY")
    print("="*50)
    print(f"Data loading time:     {load_duration:.2f}s")
    print(f"Training time:         {training_duration:.2f}s ({avg_epoch_time:.2f}s/epoch)")
    print(f"Total execution time:  {overall_duration:.2f}s")
    print("="*50)
    
    # Return model and test data
    metrics = {
        'test_loss': test_loss,
        'test_accuracy': test_accuracy,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies
    }
    
    return model, X_test, Y_test, metrics
