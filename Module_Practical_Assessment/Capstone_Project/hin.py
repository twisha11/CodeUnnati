import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow import keras
from sklearn.preprocessing import LabelBinarizer


# Define paths to the train and test directories
train_dir = 'DevanagariHandwrittenDigitDataset/Train'
test_dir = 'DevanagariHandwrittenDigitDataset/Test'

# Define the size of the images you want to load
IMG_SIZE = (32, 32)
channels = 1

# Load the train images
train_data = []
train_labels = []

for label in os.listdir(train_dir):
    path = os.path.join(train_dir, label)
    for img in os.listdir(path):
        img_path = os.path.join(path, img)
        image = cv2.imread(img_path)
        image = cv2.resize(image, IMG_SIZE)
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        train_data.append(image_gray)
        train_labels.append(label)

# Load the test images
test_data = []
test_labels = []

for label in os.listdir(test_dir):
    path = os.path.join(test_dir, label)
    for img in os.listdir(path):
        img_path = os.path.join(path, img)
        image = cv2.imread(img_path)
        image = cv2.resize(image, IMG_SIZE)
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        test_data.append(image_gray)
        test_labels.append(label)



# Convert data to numpy arrays
train_data = np.array(train_data)
test_data = np.array(test_data)

# Convert labels to numpy arrays
train_labels = np.array(train_labels)
test_labels = np.array(test_labels)

# Split the train data and labels into training and validation sets
train_data, val_data, train_labels, val_labels = train_test_split(train_data, train_labels, test_size=0.2, random_state=42)

# Preprocess the data (you can add more preprocessing steps as needed)
train_data = train_data.astype('float32') / 255.0
val_data= val_data.astype('float32') / 255.0
test_data = test_data.astype('float32') / 255.0

# Print the shapes of the data and labels
print('Train data shape:', train_data.shape)
print('Train labels shape:', train_labels.shape)
print('Validation data shape:', val_data.shape)
print('Validation labels shape:', val_labels.shape)
print('Test data shape:', test_data.shape)
print('Test labels shape:', test_labels.shape)


# Define the number of classes
NUM_CLASSES = 10

# Convert labels to one-hot encoded vectors
label_binarizer = LabelBinarizer()
train_labels = label_binarizer.fit_transform(train_labels)
val_labels = label_binarizer.transform(val_labels)
test_labels = label_binarizer.transform(test_labels)

# Define the CNN model
model = keras.models.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 1)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(128, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dropout(.5),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(.2),
    keras.layers.Dense(NUM_CLASSES, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_data, train_labels, epochs=5, batch_size=64, validation_data=(val_data, val_labels))

# Evaluate the model on the test data
test_loss, test_acc = model.evaluate(test_data, test_labels)
print('Test accuracy:', test_acc)