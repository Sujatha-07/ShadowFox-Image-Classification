
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model("models/image_classifier.keras")

# Load the CIFAR-10 test dataset
(_, _), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Normalize image pixels
x_test = x_test.astype("float32") / 255.0

# Evaluate the model
test_loss, test_accuracy = model.evaluate(
    x_test, y_test, verbose=1
)

print(f"\nTest Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
