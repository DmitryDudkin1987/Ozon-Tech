import numpy as np
import os
import tensorflow as tf


TEST_IMAGES_DIR = "./data/test/"
SUBMISSION_PATH = "./data/submission.csv"

if __name__ == "__main__":
  model = tf.keras.models.load_model('mobilenet_v2_model.keras')

all_image_names = os.listdir(TEST_IMAGES_DIR)
all_preds = []

for image_name in all_image_names:
    img_path = os.path.join(TEST_IMAGES_DIR, image_name)
    image = tf.keras.utils.load_img(img_path, target_size=(160, 160))
    input_arr = tf.keras.utils.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to a batch.
    predictions = model.predict(input_arr)
    predictions = tf.nn.sigmoid(predictions)
    predictions = tf.where(predictions < 0.75, 1, 0)
    all_preds.append(int(predictions))

with open(SUBMISSION_PATH, "w") as f:
    f.write("image_name\tlabel_id\n")
    for name, cl_id in zip(all_image_names, all_preds):
        f.write(f"{name}\t{cl_id}\n")
