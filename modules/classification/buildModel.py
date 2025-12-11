# --- Code Cell ---
import tensorflow as tf
import keras
import os
import random
from sklearn.metrics import f1_score
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.layers import Rescaling
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.metrics import Precision, Recall
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
from PIL import Image
# --- Code Cell ---
train_dir = r'C:\Stupen Celerates\Final Project\coffeBeansDataset\train'
val_dir = r'C:\Stupen Celerates\Final Project\coffeBeansDataset\val'
test_dir = r'C:\Stupen Celerates\Final Project\coffeBeansDataset\test'
# --- Code Cell ---
categories = ['defect', 'longberry', 'peaberry', 'premium']

fig, axes = plt.subplots(len(categories), 3, figsize=(12,10))
plt.suptitle("Types of Coffee Beans", fontsize=16)

for i, category in enumerate(categories):
    category_path = os.path.join(train_dir, category)
    images = os.listdir(category_path)[:3]  # ambil 3 teratas
    for j, img_name in enumerate(images):
        img_path = os.path.join(category_path, img_name)
        img = tf.keras.utils.load_img(img_path, target_size=(224,224))  # ✅ use utils
        axes[i, j].imshow(img)
        axes[i, j].set_title(category)
        axes[i, j].axis("off")

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
# --- Code Cell ---
BATCH_SIZE = 32
IMG_SIZE = 224
# --- Code Cell ---
image_gen = ImageDataGenerator(
    rescale = 1./255,
    rotation_range = 20,
    width_shift_range = 0.15,
    height_shift_range = 0.15,
    shear_range = 0.12,
    zoom_range = 0.2,
    horizontal_flip = True,
    vertical_flip = True,
    brightness_range = [0.8,1.3],
    fill_mode = 'nearest')

train_data_gen = image_gen.flow_from_directory(
    batch_size = BATCH_SIZE,
    directory = train_dir,
    shuffle = True,
    target_size = (IMG_SIZE,IMG_SIZE),
    class_mode = 'categorical',
)
val_data_gen = image_gen.flow_from_directory(
        directory = val_dir,
        batch_size = BATCH_SIZE,
        target_size = (IMG_SIZE,IMG_SIZE),
        class_mode = 'categorical',

    )
    
# --- Code Cell ---
def plotImages(images_arr):
  fig, axes = plt.subplots(1, 5, figsize = (20,20))
  axes = axes.flatten()
  for img, ax in zip(images_arr, axes):
    ax.grid(True)
    ax.imshow(img)
  plt.tight_layout()
  plt.show()
# --- Code Cell ---
augmented_images = [train_data_gen[0][0][0] for i in range(5)]
plotImages(augmented_images)
# --- Code Cell ---
sample_training_images, _ = next(train_data_gen)
# --- Code Cell ---
plotImages(sample_training_images[:5])
# --- Code Cell ---
def create_EfficientNet():
    base_model = tf.keras.applications.EfficientNetB3(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    
    # Freeze backbone untuk transfer learning tahap 1
    for layer in base_model.layers[:50]:
        layer.trainable = False

    input_layer = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base_model(input_layer, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)

    # GANTI softmax → relu (softmax tidak cocok untuk hidden layer)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.BatchNormalization()(x)

    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.2)(x)

    output_layer = layers.Dense(
        4,
        activation='softmax',
        kernel_regularizer=tf.keras.regularizers.l2(0.005)
    )(x)

    model = Model(inputs=input_layer, outputs=output_layer)

    model.compile(
        optimizer=Adam(
            learning_rate=1e-4,
            beta_1=0.9, 
            beta_2=0.999,
            epsilon=1e-07,
            amsgrad=True
        ),
        loss="categorical_crossentropy",
        metrics=["accuracy", Precision(), Recall()]
    )

    return model
# --- Code Cell ---
model = create_EfficientNet()
# --- Code Cell ---
model.summary()
# --- Code Cell ---
example_batch_images, example_batch_labels = next(iter(train_data_gen))

try:
    model.evaluate(example_batch_images, example_batch_labels, verbose = False)
except:
    print("Your Model isnt Compatible with the dataset, Check thah loss function")
else:
    predictions = model.predict(example_batch_images, verbose = False)
    print(f"Predictions's Shape : {predictions.shape}")
# --- Code Cell ---
checkpoint = ModelCheckpoint(
    filepath='best_model.keras',    
    monitor='val_loss',             
    save_best_only=True,            
    mode='min',                     
    verbose=1
)
lr_scheduler = ReduceLROnPlateau(
    monitor = 'val_accuracy',
    factor =0.5,
    patience = 3,
    min_lr=1e-07,
    verbose = 1
)
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor = 'val_loss',
    patience =7,
    restore_best_weights = True
)

with tf.device('/GPU:0'):
    history = model.fit(
        train_data_gen,
        epochs=20,
        validation_data = val_data_gen,
        callbacks = [lr_scheduler, early_stopping, checkpoint],
    )
    
# --- Code Cell ---
fig, axes = plt.subplots(1,2, figsize = (12,10))
axes[0].plot(history.history['accuracy'], label='Train Accuracy')
axes[0].plot(history.history['val_accuracy'], label='Val Accuracy')
axes[0].set_title('Accuracy over Epochs')
axes[0].set_xlabel('Epochs')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[1].plot(history.history['loss'], label='Train Loss')
axes[1].plot(history.history['val_loss'], label='Val Loss')
axes[1].set_title('Loss over Epochs')
axes[1].set_xlabel('Epochs')
axes[1].set_ylabel('Loss')
axes[1].legend()
plt.show()
