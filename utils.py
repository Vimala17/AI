import numpy as np
from PIL import Image
import io

def prepare_image(image_bytes):
    # Load image
    img = Image.open(io.BytesIO(image_bytes))
    # Resize to match the input size of your AI model (usually 224x224)
    img = img.resize((224, 224))
    # Convert to array and normalize
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array