import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

# Load the data
X_train = np.load("X_train.npy")
X_test = np.load("X_test.npy")
y_train = np.load("y_train.npy")
y_test = np.load("y_test.npy")

# Convert labels to categorical (one-hot encoding)
num_classes = len(np.unique(y_train))
"This counts how many unique gestures we have. For example, if we have 3 types of gestures, this will return 3."
y_train_cat = to_categorical(y_train, num_classes)
y_test_cat = to_categorical(y_test, num_classes)
"These lines convert the gesture labels to one-hot encoded format, which is needed for classification."

# Build the CNN model
model = Sequential([
    
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(21, 3, 1), padding='same'),
    MaxPooling2D(pool_size=(2, 1)),
    Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
    MaxPooling2D(pool_size=(2, 1)),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.4),
    Dense(num_classes, activation='softmax')
])
"This is the output layer. It has as many neurons as the number of gestures, and it uses softmax to predict probabilities for each class."

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
"This compiles the model. I’m using the Adam optimizer and categorical cross-entropy as the loss function, which is common for multi-class classification. We also track accuracy."

# Early stopping to prevent overfitting
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
"This will stop training if the validation loss doesn’t improve for 5 epochs. It also restores the best model weights."

# Train the model
model.fit(X_train, y_train_cat, epochs=30, batch_size=32,
          validation_data=(X_test, y_test_cat), callbacks=[early_stop])
"Now we train the model using the training data for up to 30 epochs. We use a batch size of 32. Validation data is used to check how the model is performing on unseen data, and early stopping is used during training."

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test_cat)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
"After training, we evaluate the model using the test data and print the accuracy."

# Save the model
model.save("gesture_cnn_model.h5")
print("Model saved as gesture_cnn_model.h5")
"Finally, we save the trained model into a file so we can use it later without training again."
