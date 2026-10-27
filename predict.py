
import sys
import numpy as np
import tensorflow as tf
from PIL import Image

class_names = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

model = tf.keras.models.load_model(
    "models/image_classifier_augmented.keras"
)

if len(sys.argv) < 2:
    print('Usage: python predict.py "path/to/image.jpg"')
    sys.exit(1)

image_path = sys.argv[1]

try:
    image = Image.open(image_path).convert("RGB")
    image = image.resize((32, 32))

    image_array = np.array(image, dtype=np.float32) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    predictions = model.predict(image_array, verbose=0)[0]
    predicted_index = int(np.argmax(predictions))

    print("Predicted class:", class_names[predicted_index])
    print(f"Confidence: {predictions[predicted_index] * 100:.2f}%")

except FileNotFoundError:
    print("Image file not found. Check the image path.")
except Exception as error:
    print("Prediction failed:", error)
