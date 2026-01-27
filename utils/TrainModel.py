import numpy as np
import os
from PIL import Image
from sklearn.model_selection import train_test_split
from Main import get_conv
from keras.layers import Flatten


def train_waldo_model(data_path='Data', epochs=15, test_size=0.10, save_path=None):
    """
    Train a model to classify Waldo vs NotWaldo images from scratch.
    
    Args:
        data_path: Path to Data directory containing Waldo/ and NotWaldo/ folders
        epochs: Number of training epochs (default 15)
        test_size: Fraction of data to use for testing (default 0.10)
        save_path: Path to save trained model weights
    """
    
    waldo_dir = os.path.join(data_path, 'Waldo')
    not_waldo_dir = os.path.join(data_path, 'NotWaldo')
    
    # Load training data
    X, Y = [], []
    
    print("Loading Waldo images (label=1)...")
    for img_file in os.listdir(waldo_dir):
        if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(waldo_dir, img_file)
            try:
                img_array = np.array(Image.open(img_path))
                X.append(img_array)
                Y.append(1)
            except Exception as e:
                print(f"Error loading {img_file}: {e}")
    
    print("Loading NotWaldo images (label=0)...")
    for img_file in os.listdir(not_waldo_dir):
        if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(not_waldo_dir, img_file)
            try:
                img_array = np.array(Image.open(img_path))
                X.append(img_array)
                Y.append(0)
            except Exception as e:
                print(f"Error loading {img_file}: {e}")
    
    # Convert to numpy arrays
    X = np.array(X)
    Y = np.array(Y)
    
    print(f"\nTotal images: {len(X)}")
    print(f"Waldo images: {sum(Y)}, NotWaldo images: {len(Y) - sum(Y)}")
    
    # Split data
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=42)
    
    print(f'\nX_train shape: {X_train.shape}')
    print(f'X_test shape: {X_test.shape}')
    
    # Create and train model from scratch (no pre-trained weights)
    model = get_conv()  # Create fresh model architecture
    model.add(Flatten())
    model.compile(loss='mse', optimizer='adadelta', metrics=['accuracy'])
    
    print("\nTraining model...")
    model.fit(X_train, Y_train, batch_size=32, epochs=epochs, verbose=1, validation_data=(X_test, Y_test))
    
    # Evaluate
    score = model.evaluate(X_test, Y_test, verbose=0)
    print(f'\nTest score: {score[0]:.4f}')
    print(f'Test accuracy: {score[1]:.4f}')
    
    # Save trained weights
    model.save_weights(save_path)
    print(f"\nModel saved to {save_path}")
    
    return model, X_test, Y_test, score
