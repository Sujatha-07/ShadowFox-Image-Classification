
# import os
# import tensorflow as tf
# import numpy as np
# import matplotlib.pyplot as plt

# # Make results reproducible
# tf.random.set_seed(42)
# np.random.seed(42)

# # Create folders for saved model and outputs
# os.makedirs("models", exist_ok=True)
# os.makedirs("outputs", exist_ok=True)

# # CIFAR-10 image categories
# class_names = [
#     "airplane", "automobile", "bird", "cat", "deer",
#     "dog", "frog", "horse", "ship", "truck"
# ]

# # Load the dataset
# (x_train, y_train), (x_test, y_test) = (
#     tf.keras.datasets.cifar10.load_data()
# )

# # Normalize image pixels from 0–255 to 0–1
# x_train = x_train.astype("float32") / 255.0
# x_test = x_test.astype("float32") / 255.0

# y_train = y_train.flatten()
# y_test = y_test.flatten()

# print("Training images:", x_train.shape)
# print("Testing images:", x_test.shape)

# # Build a convolutional neural network
# model = tf.keras.Sequential([
#     tf.keras.layers.Input(shape=(32, 32, 3)),

#     tf.keras.layers.Conv2D(
#         32, (3, 3), activation="relu", padding="same"
#     ),
#     tf.keras.layers.MaxPooling2D((2, 2)),

#     tf.keras.layers.Conv2D(
#         64, (3, 3), activation="relu", padding="same"
#     ),
#     tf.keras.layers.MaxPooling2D((2, 2)),

#     tf.keras.layers.Conv2D(
#         128, (3, 3), activation="relu", padding="same"
#     ),
#     tf.keras.layers.GlobalAveragePooling2D(),

#     tf.keras.layers.Dense(64, activation="relu"),
#     tf.keras.layers.Dropout(0.3),
#     tf.keras.layers.Dense(10, activation="softmax")
# ])

# model.compile(
#     optimizer="adam",
#     loss="sparse_categorical_crossentropy",
#     metrics=["accuracy"]
# )

# model.summary()

# # Train the model
# history = model.fit(
#     x_train,
#     y_train,
#     epochs=10,
#     batch_size=64,
#     validation_split=0.1
# )

# # Evaluate on unseen test images
# test_loss, test_accuracy = model.evaluate(
#     x_test, y_test, verbose=1
# )

# print(f"Test accuracy: {test_accuracy * 100:.2f}%")

# # Save the trained model
# model.save("models/image_classifier.keras")

# # Save the training curves
# plt.figure(figsize=(10, 4))

# plt.subplot(1, 2, 1)
# plt.plot(history.history["accuracy"], label="Training")
# plt.plot(history.history["val_accuracy"], label="Validation")
# plt.title("Model Accuracy")
# plt.xlabel("Epoch")
# plt.ylabel("Accuracy")
# plt.legend()

# plt.subplot(1, 2, 2)
# plt.plot(history.history["loss"], label="Training")
# plt.plot(history.history["val_loss"], label="Validation")
# plt.title("Model Loss")
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.legend()

# plt.tight_layout()
# plt.savefig("outputs/training_curves.png")
# plt.close()

# print("Model saved to models/image_classifier.keras")
# print("Training curves saved to outputs/training_curves.png")

import os
import tensorflow as tf
import matplotlib.pyplot as plt

# Reproducibility
tf.keras.utils.set_random_seed(42)

# Create output folders
os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = (
    tf.keras.datasets.cifar10.load_data()
)

# Normalize pixel values to the range 0–1
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Data augmentation: applied during training only
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.05),
    tf.keras.layers.RandomTranslation(0.1, 0.1),
    tf.keras.layers.RandomZoom(0.1),
], name="data_augmentation")

# Build the improved CNN
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(32, 32, 3)),

    data_augmentation,

    tf.keras.layers.Conv2D(
        32, (3, 3), padding="same", activation="relu"
    ),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(
        32, (3, 3), padding="same", activation="relu"
    ),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Conv2D(
        64, (3, 3), padding="same", activation="relu"
    ),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(
        64, (3, 3), padding="same", activation="relu"
    ),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Conv2D(
        128, (3, 3), padding="same", activation="relu"
    ),
    tf.keras.layers.GlobalAveragePooling2D(),

    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(10, activation="softmax"),
])

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# Stop if validation accuracy stops improving
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    patience=4,
    mode="max",
    restore_best_weights=True,
)

# Reduce learning rate when validation accuracy plateaus
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=2,
    min_lr=0.00001,
    verbose=1,
)

# Train the model
history = model.fit(
    x_train,
    y_train,
    epochs=20,
    batch_size=64,
    validation_split=0.1,
    callbacks=[early_stopping, reduce_lr],
)

# Evaluate on the separate test dataset
test_loss, test_accuracy = model.evaluate(
    x_test, y_test, verbose=1
)

print(f"\nTest Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Save to a new file so the existing model remains safe
model.save("models/image_classifier_augmented.keras")

# Plot training and validation curves
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Training")
plt.plot(history.history["val_accuracy"], label="Validation")
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Training")
plt.plot(history.history["val_loss"], label="Validation")
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.savefig("outputs/training_curves_augmented.png")
plt.show()

print("\nImproved model saved successfully!")
