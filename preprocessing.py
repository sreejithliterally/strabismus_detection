
import os
import cv2
import numpy as np

# Define directories for normal and strabised images
normal_dir = 'normal'
strabised_dir = 'strabised'

# Function to preprocess images
def preprocess_image(image_path, target_size=(224, 224)):
    # Read image
    image = cv2.imread(image_path)
    # Convert to grayscale if necessary
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Resize image
    image = cv2.resize(image, target_size)
    # Normalize pixel values
    image = image / 255.0
    return image

# Function to load images from directories and preprocess them
def load_and_preprocess_images(directory):
    images = []
    labels = []
    # Iterate through files in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
            # Preprocess each image and append to the list
            image_path = os.path.join(directory, filename)
            try:
                preprocessed_image = preprocess_image(image_path)
                images.append(preprocessed_image)
                # Add label based on directory name
                if directory == normal_dir:
                    labels.append(0)  # 0 for normal
                elif directory == strabised_dir:
                    labels.append(1)  # 1 for strabised
            except Exception as e:
                print(f"Error processing image {filename}: {str(e)}")
    return images, labels

# Load and preprocess normal and strabised images
normal_images, normal_labels = load_and_preprocess_images(normal_dir)
strabised_images, strabised_labels = load_and_preprocess_images(strabised_dir)

# Convert lists to NumPy arrays for further processing
normal_images = np.array(normal_images)
normal_labels = np.array(normal_labels)
strabised_images = np.array(strabised_images)
strabised_labels = np.array(strabised_labels)

# Combine normal and strabised images and labels
images = np.concatenate((normal_images, strabised_images), axis=0)
labels = np.concatenate((normal_labels, strabised_labels), axis=0)

# Shuffle the data
shuffle_indices = np.random.permutation(len(images))
images = images[shuffle_indices]
labels = labels[shuffle_indices]

# Display the shape of the arrays and number of samples per class
print("Images shape:", images.shape)
print("Labels shape:", labels.shape)
print("Number of normal images:", np.sum(labels == 0))
print("Number of strabised images:", np.sum(labels == 1))

# Print all values in the images array
print("Values in images array:")
print(images)

# Print all values in the labels array
print("\nValues in labels array:")
print(labels)
