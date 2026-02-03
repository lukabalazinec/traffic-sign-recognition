import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os

np.random.seed(42)
tf.random.set_seed(42)

def load_data():
    train_df = pd.read_csv('Train.csv')
    
    images = []
    labels = []
    
    for _, row in train_df.iterrows():
        img_path = row['Path']
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            x1, y1, x2, y2 = row['Roi.X1'], row['Roi.Y1'], row['Roi.X2'], row['Roi.Y2']
            
            img = img[y1:y2, x1:x2]
            
            img = cv2.resize(img, (64, 64))
            
            img = img / 255.0
            
            images.append(img)
            labels.append(row['ClassId'])
    
    X = np.array(images)
    y = np.array(labels)
    
    return X, y

def prepare_data():
    X, y = load_data()
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    num_classes = len(np.unique(y))
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_val = keras.utils.to_categorical(y_val, num_classes)
    
    return X_train, X_val, y_train, y_val, num_classes

class ProgressiveDataGenerator(keras.utils.Sequence):
    def __init__(self, x, y, batch_size=32):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.current_epoch = 0
        
        # Base augmentation layers
        self.rotation = keras.layers.RandomRotation(0.15)
        self.zoom = keras.layers.RandomZoom(0.1)
        self.brightness = keras.layers.RandomBrightness(0.2)
        
    def __len__(self):
        return int(np.ceil(len(self.x) / self.batch_size))
    
    def on_epoch_end(self):
        self.current_epoch += 1
    
    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        batch_x = tf.cast(batch_x, tf.float32)
        
        if self.current_epoch >= 12:
            batch_x = self.rotation(batch_x)
            
        if self.current_epoch >= 26:
            batch_x = self.zoom(batch_x)
            
        if self.current_epoch >= 40:
            batch_x = self.brightness(batch_x)
        
        return batch_x, batch_y

def create_model(num_classes):
    inputs = keras.Input(shape=(64, 64, 3))
    
    conv1 = keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
    conv1 = keras.layers.BatchNormalization()(conv1)
    conv1 = keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu')(conv1)
    conv1 = keras.layers.BatchNormalization()(conv1)
    pool1 = keras.layers.MaxPooling2D((2, 2))(conv1)
    
    conv2 = keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')(pool1)
    conv2 = keras.layers.BatchNormalization()(conv2)
    conv2 = keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')(conv2)
    conv2 = keras.layers.BatchNormalization()(conv2)
    pool2 = keras.layers.MaxPooling2D((2, 2))(conv2)
    
    conv3 = keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu')(pool2)
    conv3 = keras.layers.BatchNormalization()(conv3)
    conv3 = keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu')(conv3)
    conv3 = keras.layers.BatchNormalization()(conv3)
    pool3 = keras.layers.MaxPooling2D((2, 2))(conv3)
    
    gap = keras.layers.GlobalAveragePooling2D()(pool3)
    
    dense1 = keras.layers.Dense(512, activation='relu')(gap)
    dense1 = keras.layers.BatchNormalization()(dense1)
    dense1 = keras.layers.Dropout(0.5)(dense1)
    
    dense2 = keras.layers.Dense(256, activation='relu')(dense1)
    dense2 = keras.layers.BatchNormalization()(dense2)
    dense2 = keras.layers.Dropout(0.3)(dense2)
    
    outputs = keras.layers.Dense(num_classes, activation='softmax')(dense2)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    return model

def train_model(model, X_train, y_train, X_val, y_val):
    initial_learning_rate = 0.001
    
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=1000,
        decay_rate=0.9,
        staircase=True
    )
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    train_generator = ProgressiveDataGenerator(X_train, y_train, batch_size=32)
    val_generator = ProgressiveDataGenerator(X_val, y_val, batch_size=32)
    
    callbacks = [
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.00001,
            min_delta=0.001
        ),
        keras.callbacks.ModelCheckpoint(
            'best_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )
    ]
    
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=50,
        callbacks=callbacks
    )
    
    return history

def plot_training_history(history):
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("Loading and preparing data...")
    X_train, X_val, y_train, y_val, num_classes = prepare_data()
    print(f"Training set shape: {X_train.shape}")
    print(f"Validation set shape: {X_val.shape}")
    print(f"Number of classes: {num_classes}")
    
    print("\nCreating and training the model...")
    model = create_model(num_classes)
    model.summary()
    
    history = train_model(model, X_train, y_train, X_val, y_val)
    
    plot_training_history(history)
    
    model.save('traffic_sign_model.h5')
    print("\nModel saved as 'traffic_sign_model.h5'")
