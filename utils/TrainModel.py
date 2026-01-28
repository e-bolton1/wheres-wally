import numpy as np
import os
import time
import gc
from PIL import Image
from sklearn.model_selection import train_test_split
from Main import get_conv
from keras.layers import Flatten, Dense
import keras.backend as K


def train_waldo_model(data_path='Data', epochs=15, test_size=0.10, save_path=None, img_size=64):
    """
    Train a model to classify Waldo vs NotWaldo images from scratch.
    
    Args:
        data_path: Path to Data directory containing Waldo/ and NotWaldo/ folders
        epochs: Number of training epochs (default 15)
        test_size: Fraction of data to use for testing (default 0.10)
        save_path: Path to save trained model weights
        img_size: Size to resize all images to (default 64)
    """
    
    # Start overall timer
    overall_start_time = time.time()
    
    # Clear memory before starting
    gc.collect()
    K.clear_session()
    
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
                img_array = np.array(img, dtype=np.float32) / 255.0  # Normalize to [0,1]
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
                img_array = np.array(img, dtype=np.float32) / 255.0  # Normalize to [0,1]
                X.append(img_array)
                Y.append(0)
            except Exception as e:
                print(f"Error loading {img_file}: {e}")
    
    # Convert to numpy arrays
    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.float32)
    
    # Clear temporary memory
    gc.collect()
    
    load_end_time = time.time()
    load_duration = load_end_time - load_start_time
    
    print(f"\nTotal images: {len(X)}")
    print(f"Waldo images: {sum(Y)}, NotWaldo images: {len(Y) - sum(Y)}")
    print(f"Data loading time: {load_duration:.2f} seconds")
    
    # Split data
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=42)
    
    # Free up the original arrays
    del X, Y
    gc.collect()
    
    print(f'\nX_train shape: {X_train.shape}')
    print(f'X_test shape: {X_test.shape}')
    
    # Create and train model from scratch (no pre-trained weights)
    model = get_conv(input_shape=(img_size, img_size, 3), filename=None)  # Use fixed image size
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))  # Binary classification output
    model.compile(loss='binary_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    
    print("\nTraining model...")
    training_start_time = time.time()
    # Reduced batch size from 32 to 16 to reduce memory usage
    model.fit(X_train, Y_train, batch_size=16, epochs=epochs, verbose=1, validation_data=(X_test, Y_test))
    training_end_time = time.time()
    training_duration = training_end_time - training_start_time
    
    # Evaluate
    score = model.evaluate(X_test, Y_test, verbose=0)
    print(f'\nTest score: {score[0]:.4f}')
    print(f'Test accuracy: {score[1]:.4f}')
    
    # Save trained weights
    model.save_weights(save_path)
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
    
    return model, X_test, Y_test, score
