import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import json

# Setting paths
base_path = 'nutritionverse-data'
images_path = os.path.join(base_path, 'nutritionverse-manual/nutritionverse-manual/images')
metadata_path = os.path.join(base_path, 'nutritionverse_dish_metadata3.csv')
coco_json_path = os.path.join(base_path, 'nutritionverse-manual/nutritionverse-manual/images/_annotations.coco.json')
splits_path = os.path.join(base_path, 'nutritionverse-manual/nutritionverse-manual/updated-manual-dataset-splits.csv')

# Load COCO format image metadata
with open(coco_json_path, 'r') as f:
    coco_data = json.load(f)

print(f"COCO metadata loaded with {len(coco_data['images'])} images and {len(coco_data['annotations'])} annotations")

# Optionally extract image information from COCO format
coco_images = pd.DataFrame(coco_data['images'])
print("COCO images structure:")
print(coco_images.columns.tolist())

# Create a mapping from image_id to file_name
image_id_to_filename = {img['id']: img['file_name'] for img in coco_data['images']}
print(f"Created mapping for {len(image_id_to_filename)} image IDs to filenames")

# Load dataset splits if available
if os.path.exists(splits_path):
    splits_df = pd.read_csv(splits_path)
    print(f"Dataset splits loaded with {len(splits_df)} entries")
    print(f"Split distribution: {splits_df['category'].value_counts().to_dict()}")
else:
    print("Dataset splits file not found. Will use random splitting.")
    splits_df = None

# Load metadata
metadata_df = pd.read_csv(metadata_path)

# Display first few rows of metadata
print("Dataset Metadata Overview:")
print(f"Total entries: {len(metadata_df)}")
metadata_df.head()


# add category to annotation df
annotation = coco_data['annotations']
category = coco_data['categories']
#each annotation consist of id, image_id, category_id, bbox, area, and segmentation(array of array )

for a in annotation:
    a['category'] = category[a['category_id']]['name']


# Import necessary libraries for image processing and model building
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import cv2
import numpy as np
from tqdm import tqdm
import random
import pandas as pd

# Define constants
IMG_SIZE = 224  # Input size for MobileNetV2
BATCH_SIZE = 32
EPOCHS = 15
NUM_CLASSES = len(coco_data['categories'])

# Function to extract and preprocess food regions based on segmentation
def extract_food_regions(image_ids, coco_data, image_id_to_filename, images_path):
    X = []  # Cropped food images
    y = []  # Food category labels
    
    progress_bar = tqdm(image_ids)
    for img_id in progress_bar:
        try:
            # Get the filename for this image ID
            filename = image_id_to_filename[img_id]
            img_path = os.path.join(images_path, filename)
            
            # Load the image
            image = cv2.imread(img_path)
            if image is None:
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Get annotations for this image
            annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] == img_id]
            
            for ann in annotations:
                category_id = ann['category_id']
                bbox = ann['bbox']  # [x, y, width, height]
                
                # Extract region using bounding box
                x, y_coord, w, h = [int(v) for v in bbox]
                
                # Ensure coordinates are within image boundaries
                x = max(0, x)
                y_coord = max(0, y_coord)
                w = min(w, image.shape[1] - x)
                h = min(h, image.shape[0] - y_coord)
                
                if w <= 0 or h <= 0:
                    continue
                
                # Extract the region
                food_region = image[y_coord:y_coord+h, x:x+w]
                
                # Resize to standard size
                food_region = cv2.resize(food_region, (IMG_SIZE, IMG_SIZE))
                
                # Normalize pixel values
                food_region = food_region / 255.0
                
                # Add to dataset
                X.append(food_region)
                y.append(category_id)
                
        except Exception as e:
            print(f"Error processing image {img_id}: {e}")
    
    return np.array(X), np.array(y)


# Read the CSV file with train-test split information
print("Loading train-test split from CSV file...")
try:
    split_df = pd.read_csv('nutritionverse-data/nutritionverse-manual/nutritionverse-manual/updated-manual-dataset-splits.csv', header=None)
    split_df.columns = ['file_name', 'category']
    
    # Create dictionaries for train and test splits
    train_filenames = set(split_df[split_df['split'] == 'Train']['file_name'])
    test__filenames = set(split_df[split_df['split'] == 'Test']['file_name'])
    
    print(f"Found {len(train_filenames)} training images and {len(test__filenames)} test images in the split file")
except Exception as e:
    print(f"Error loading split file: {e}")
    print("Will fall back to random train-test split")
    train_filenames = None
    test__filenames = None

# Function to split the dataset and extract food regions
def prepare_train_test_data(coco_data, image_id_to_filename, images_path, train_filenames, test_filenames):
    # Get all image IDs
    all_image_ids = [img['id'] for img in coco_data['images']]
    
    # Get train and test image IDs based on predefined splits
    if train_filenames is not None and test_filenames is not None:
        train_image_ids = [img_id for img_id in all_image_ids if image_id_to_filename[img_id] in train_filenames]
        test_image_ids = [img_id for img_id in all_image_ids if image_id_to_filename[img_id] in test_filenames]
    else:
        # If no predefined splits, randomly split the dataset
        random.shuffle(all_image_ids)
        split_index = int(0.8 * len(all_image_ids))  # 80% for training, 20% for testing
        train_image_ids = all_image_ids[:split_index]
        test_image_ids = all_image_ids[split_index:]
    
    print(f"Using {len(train_image_ids)} training images and {len(test_image_ids)} testing images.")
    
    # Extract food regions for training and testing datasets
    print("Extracting food regions for training dataset...")
    X_train, y_train = extract_food_regions(train_image_ids, coco_data, image_id_to_filename, images_path)
    
    print("Extracting food regions for testing dataset...")
    X_test, y_test = extract_food_regions(test_image_ids, coco_data, image_id_to_filename, images_path)
    
    return X_train, y_train, X_test, y_test

# Prepare the train and test datasets
X_train, y_train, X_test, y_test = prepare_train_test_data(
    coco_data, image_id_to_filename, images_path, train_filenames, test__filenames
)

# Print dataset shapes
print(f"Training dataset: {X_train.shape}, {y_train.shape}")
print(f"Testing dataset: {X_test.shape}, {y_test.shape}")

import logging
import time

TIMESTAMP = int(time.time())
# Create a dynamic log file name
log_file_name = f"training_log_E{EPOCHS}_B{BATCH_SIZE}_IMG{IMG_SIZE}_TIMEP{TIMESTAMP}.log"

# Configure logging
logging.basicConfig(
    filename=log_file_name,  # Use the dynamic log file name
    level=logging.INFO,  # Log level
    format='%(asctime)s - %(levelname)s - %(message)s'  # Log format
)

# Custom callback to log training progress
class TrainingLogger(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logging.info(
            f"Epoch {epoch + 1}: "
            f"loss = {logs.get('loss'):.4f}, "
            f"accuracy = {logs.get('accuracy'):.4f}, "
            f"val_loss = {logs.get('val_loss'):.4f}, "
            f"val_accuracy = {logs.get('val_accuracy'):.4f}"
        )
        print(
            f"Epoch {epoch + 1}: "
            f"loss = {logs.get('loss'):.4f}, "
            f"accuracy = {logs.get('accuracy'):.4f}, "
            f"val_loss = {logs.get('val_loss'):.4f}, "
            f"val_accuracy = {logs.get('val_accuracy'):.4f}"
        )

# Build a CNN model for food detection
def build_food_detection_model(num_classes):
    
    model = models.Sequential([
        # First convolutional block
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        layers.MaxPooling2D((2, 2)),

        # Second convolutional block
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        # Third convolutional block
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        # Fourth convolutional block
        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Flatten and fully connected layers
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.1),
        layers.Dense(num_classes, activation='softmax')
    ])

    return model


# Create and compile the model
model = build_food_detection_model(NUM_CLASSES)
model.compile(
    optimizer=optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Display model summary
model.summary()

# Data augmentation for training
datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Train the model
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
    validation_data=(X_test, y_test),
    epochs=EPOCHS,
    steps_per_epoch=len(X_train) // BATCH_SIZE,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=7, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.1, patience=2),
        TrainingLogger()
    ]
)

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc:.4f}")

# Plot training history
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

model.save("raw_4layer32_224Size_32bs_15E.h5")

# import tensorflow as tf
# from tensorflow.keras import layers, models, optimizers
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# import numpy as np
# import cv2
# import pandas as pd
# import random
# from sklearn.preprocessing import LabelEncoder

# def detect_food_in_image(image_path, model, label_encoder):
#     # Load and preprocess the image
#     image = cv2.imread(image_path)
#     if image is None:
#         return "Image not found."
    
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     original_image = image.copy()
    
#     # For visualization
#     plt.figure(figsize=(12, 12))
#     plt.imshow(original_image)
#     plt.title("Food Detection Results")
    
#     results = []
    
#     # Method 1: Contour-based detection
#     # Convert to HSV color space for better color segmentation
#     hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    
#     # Apply Gaussian blur to reduce noise
#     blurred = cv2.GaussianBlur(hsv_image, (5, 5), 0)
    
#     # Use adaptive thresholding to find potential regions of interest
#     gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
#     thresh = cv2.adaptiveThreshold(
#         gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
#         cv2.THRESH_BINARY_INV, 11, 2
#     )
    
#     # Find contours in the thresholded image
#     contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
#     # Filter contours by area to remove tiny regions
#     min_area = image.shape[0] * image.shape[1] * 0.01  # 1% of image area
#     significant_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    
#     # Analyze each significant contour
#     for i, contour in enumerate(significant_contours):
#         # Get bounding rectangle
#         x, y, w, h = cv2.boundingRect(contour)
        
#         # Ensure the region is reasonably sized
#         if w < 20 or h < 20:
#             continue
            
#         # Extract the region
#         food_region = original_image[y:y+h, x:x+w]
        
#         # Resize to standard size
#         food_region_resized = cv2.resize(food_region, (IMG_SIZE, IMG_SIZE))
        
#         # Normalize pixel values
#         food_region_normalized = food_region_resized / 255.0
        
#         # Predict
#         prediction = model.predict(np.expand_dims(food_region_normalized, axis=0), verbose=0)
#         predicted_class_idx = np.argmax(prediction)
#         confidence = prediction[0][predicted_class_idx]
        
#         # Filter predictions with low confidence
#         if confidence < 0.4:
#             continue
            
#         # Convert to category name
#         predicted_class = label_encoder.inverse_transform([predicted_class_idx])[0]
#         category_name = next((cat['name'] for cat in coco_data['categories'] if cat['id'] == predicted_class), "Unknown")
        
#         # Add to results
#         results.append({
#             'category': category_name,
#             'confidence': float(confidence),
#             'bbox': [x, y, w, h]
#         })
        
#         # Draw rectangle and label on the image
#         rect = plt.Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none')
#         plt.gca().add_patch(rect)
#         plt.text(x, y - 5, f"{category_name} ({confidence:.2f})", 
#                  color='white', fontsize=12, backgroundcolor='red')
    
#     # Method 2: Multi-scale sliding window detection
#     # Run regardless of whether contours found anything, to improve detection of multiple foods
#     print("Running sliding window detection for multiple food items...")
    
#     # Define window sizes (relative to image size)
#     window_sizes = [
#         (image.shape[1]//2, image.shape[0]//2),  # Half image size
#         (image.shape[1]//3, image.shape[0]//3),  # One-third size
#         # (image.shape[1]//4, image.shape[0]//4)   # One-fourth size
#     ]
    
#     # Smaller step size to catch more food items (with overlap)
#     step_size = min(image.shape[0], image.shape[1]) // 8
    
#     # Track detection regions to avoid duplicates
#     detection_heatmap = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
#     sliding_window_results = []
    
#     # Process each window size
#     for win_w, win_h in window_sizes:
#         # Skip if window is too small
#         if win_w < 50 or win_h < 50:
#             continue
            
#         for y in range(0, image.shape[0] - win_h, step_size):
#             for x in range(0, image.shape[1] - win_w, step_size):
#                 # Extract window
#                 window = original_image[y:y+win_h, x:x+win_w]
                
#                 # Resize to standard size
#                 window_resized = cv2.resize(window, (IMG_SIZE, IMG_SIZE))
                
#                 # Normalize pixel values
#                 window_normalized = window_resized / 255.0
                
#                 # Predict
#                 prediction = model.predict(np.expand_dims(window_normalized, axis=0), verbose=0)
#                 predicted_class_idx = np.argmax(prediction)
#                 confidence = prediction[0][predicted_class_idx]
                
#                 # Filter predictions with low confidence
#                 if confidence < 0.5:
#                     continue
                    
#                 # Convert to category name
#                 predicted_class = label_encoder.inverse_transform([predicted_class_idx])[0]
#                 category_name = next((cat['name'] for cat in coco_data['categories'] if cat['id'] == predicted_class), "Unknown")
                
#                 # Add to sliding window results
#                 sliding_window_results.append({
#                     'category': category_name,
#                     'confidence': float(confidence),
#                     'bbox': [x, y, win_w, win_h]
#                 })
                
#                 # Add to heatmap for later NMS
#                 detection_heatmap[y:y+win_h, x:x+win_w] += confidence
    
#     # Non-maximum suppression to eliminate overlapping windows
#     # Sort sliding window results by confidence (highest first)
#     sliding_window_results.sort(key=lambda x: x['confidence'], reverse=True)
    
#     # Keep track of which results to keep
#     keep_results = []
    
#     # Simple NMS implementation
#     for i, current in enumerate(sliding_window_results):
#         # Skip if this window was already suppressed
#         if current in keep_results:
#             continue
            
#         # Current window coordinates
#         x1, y1, w1, h1 = current['bbox']
#         x2, y2 = x1 + w1, y1 + h1
        
#         # Keep this window
#         keep_results.append(current)
        
#         # Check against all other windows
#         for j, other in enumerate(sliding_window_results):
#             if i == j or other in keep_results:
#                 continue
                
#             # Other window coordinates
#             x3, y3, w2, h2 = other['bbox']
#             x4, y4 = x3 + w2, y3 + h2
            
#             # Calculate intersection area
#             x_overlap = max(0, min(x2, x4) - max(x1, x3))
#             y_overlap = max(0, min(y2, y4) - max(y1, y3))
#             intersection = x_overlap * y_overlap
            
#             # Calculate union area
#             area1 = w1 * h1
#             area2 = w2 * h2
#             union = area1 + area2 - intersection
            
#             # Calculate IoU (Intersection over Union)
#             iou = intersection / union if union > 0 else 0
            
#             # If there's significant overlap and same category, suppress the lower confidence window
#             if iou > 0.5 and current['category'] == other['category']:
#                 # Don't add other to keep_results - it's being suppressed
#                 pass
    
#     # Add the filtered sliding window results to the overall results
#     for result in keep_results:
#         x, y, w, h = result['bbox']
#         category_name = result['category']
#         confidence = result['confidence']
        
#         # Add to results if not already detected by contour method
#         # (Simple check - if there's no major overlap with existing results)
#         should_add = True
#         for existing in results:
#             ex, ey, ew, eh = existing['bbox']
#             if (x < ex + ew and x + w > ex and 
#                 y < ey + eh and y + h > ey and
#                 existing['category'] == category_name):
#                 should_add = False
#                 break
                
#         if should_add:
#             results.append(result)
            
#             # Draw rectangle and label on the image
#             rect = plt.Rectangle((x, y), w, h, linewidth=2, edgecolor='b', facecolor='none')
#             plt.gca().add_patch(rect)
#             plt.text(x, y - 5, f"{category_name} ({confidence:.2f})", 
#                      color='white', fontsize=12, backgroundcolor='blue')
    
#     # Display results
#     plt.axis('off')
#     plt.show()
    
#     print(f"Detected {len(results)} food items in total.")
#     return results

# category = coco_data['categories']
# # Extract only the 'id' values from the categories
# category_ids = [cat['id'] for cat in category]

# # Initialize and fit the LabelEncoder
# label_encoder = LabelEncoder()
# label_encoder.fit(category_ids)

# # Test the detection on a sample image
# sample_image_id = random.choice(list(image_id_to_filename.keys()))
# sample_filename = image_id_to_filename[sample_image_id]
# sample_image_path = os.path.join(images_path, sample_filename)

# print(f"Testing food detection on image: {sample_filename}")

# # Predict food categories
# detected_foods = detect_food_in_image(sample_image_path, model, label_encoder)

# print("Detected food items:")
# for i, food in enumerate(detected_foods):
#     print(f"{i+1}. {food['category']} (Confidence: {food['confidence']:.2f})")