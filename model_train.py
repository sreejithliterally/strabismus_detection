import numpy as np
from preprocessing import images, labels

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.callbacks import ModelCheckpoint

print("Data loading started...")
# Load preprocessed data from preprocessing.py
# Make sure to adjust the import statements accordingly
print("Data loading complete.")

print("Data splitting started...")
# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42, stratify=labels)
print("Data splitting complete.")

# Define the CNN architecture
print("Model building started...")
model = Sequential()
# Add convolutional layers, max pooling, and dense layers as needed
# Example architecture:
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
print("Model building complete.")

# Compile the model
print("Model compilation started...")
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print("Model compilation complete.")

# Define callbacks (e.g., ModelCheckpoint to save the best model)
print("Callbacks definition complete.")

# Train the model
print("Model training started...")
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val), callbacks=[checkpoint])
print("Model training complete.")

# Save the trained model
print("Saving trained model...")
model.save('final_model.h5')
print("Trained model saved.")
