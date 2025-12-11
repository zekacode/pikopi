# --- Code Cell ---
from tensorflow import keras

model = keras.models.load_model("best_model.keras")

# --- Code Cell ---
from tensorflow.keras.preprocessing.image import ImageDataGenerator

test_data_gen = ImageDataGenerator(rescale = 1.0/255.0)

BATCH_SIZE = 32
IMG_SIZE = 224
# --- Code Cell ---
test_data_gen = test_data_gen.flow_from_directory(
    directory="input/Beans/test",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=224,
    class_mode="categorical",
)

# --- Code Cell ---
model.evaluate(test_data_gen)
