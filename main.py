import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Loading training dataset
df = pd.read_csv("data/train/train.csv")
train_filepaths = df["Image"].values
train_labelpaths = df["Label"].values
ds_train = tf.data.Dataset.from_tensor_slices((train_filepaths, train_labelpaths))

# Loading testing dataset
df = pd.read_csv("data/test/test.csv")
test_filepaths = df["Image"].values
test_labelpaths = df["Label"].values
ds_test = tf.data.Dataset.from_tensor_slices((test_filepaths, test_labelpaths))

# Convert image format to tensor
def read_image(image_file, label_file):
    image = tf.io.read_file(image_file)
    image = tf.image.decode_image(image, channels = 1, dtype = tf.float32)
    image.set_shape([28, 28, 1]) # Fix as_list() bug
    label = tf.io.read_file(label_file)
    label = tf.image.decode_image(label, channels = 1, dtype = tf.float32)
    label.set_shape([28, 28, 1]) # Fix as_list() bug
    return image, label

AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 64

ds_train = ds_train.map(read_image, num_parallel_calls = AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(len(train_filepaths))
ds_train = ds_train.batch(BATCH_SIZE)
ds_train = ds_train.prefetch(AUTOTUNE)

ds_test = ds_test.map(read_image, num_parallel_calls = AUTOTUNE)
ds_test = ds_test.batch(BATCH_SIZE)
ds_test = ds_test.prefetch(AUTOTUNE)

# Functional API Model
inputs = keras.Input(shape = (None, None, 1))
x = layers.Conv2D(32, 3, padding = "same")(inputs)
x = layers.BatchNormalization()(x)
x = keras.activations.relu(x)
x = layers.Conv2D(64, 3, padding = "same")(x)
x = layers.BatchNormalization()(x)
x = keras.activations.relu(x)
x = layers.Conv2D(64, 3, padding = "same")(x)
x = layers.BatchNormalization()(x)
x = keras.activations.relu(x)
x = layers.MaxPooling2D()(x)

x = layers.Conv2D(64, 5, padding = "same")(x)
x = layers.BatchNormalization()(x)
x = keras.activations.relu(x)
x = layers.Conv2D(128, 5, padding = "same")(x)
x = layers.BatchNormalization()(x)
x = keras.activations.relu(x)
x = layers.Conv2D(128, 5, padding = "same")(x)
x = layers.BatchNormalization()(x)
x = keras.activations.relu(x)
x = layers.MaxPooling2D()(x)

x = layers.UpSampling2D()(x)
x = layers.Conv2DTranspose(128, 5, padding = "same")(x)
x = layers.BatchNormalization()(x)
x = keras.activations.relu(x)
x = layers.Dropout(0.5)(x)
x = layers.Conv2DTranspose(128, 5, padding = "same")(x)
x = layers.BatchNormalization()(x)
x = keras.activations.relu(x)
x = layers.Dropout(0.5)(x)
x = layers.Conv2DTranspose(64, 5, padding = "same")(x)
x = layers.BatchNormalization()(x)
x = keras.activations.relu(x)
x = layers.Dropout(0.5)(x)

x = layers.UpSampling2D()(x)
x = layers.Conv2DTranspose(64, 3, padding = "same")(x)
x = layers.BatchNormalization()(x)
x = keras.activations.relu(x)
x = layers.Dropout(0.5)(x)
x = layers.Conv2DTranspose(64, 3, padding = "same")(x)
x = layers.BatchNormalization()(x)
x = keras.activations.relu(x)
x = layers.Dropout(0.5)(x)
x = layers.Conv2DTranspose(32, 3, padding = "same")(x)
x = layers.BatchNormalization()(x)
x = keras.activations.relu(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Conv2DTranspose(1, 3, padding = "same", activation = "relu")(x)

model = keras.Model(inputs = inputs, outputs = outputs)

model.compile(
    optimizer = keras.optimizers.Adam(),
    loss = keras.losses.BinaryCrossentropy(from_logits = False),
    metrics = ["accuracy"]
)
print(model.summary())
#print(model.trainable_variables)

model.fit(ds_train, epochs = 1, verbose = 1)
model.evaluate(ds_test, verbose = 1)
model.save_weights("checkpoint/")

for i in range(10):
    test_image = tf.io.read_file("pred_images/" + str(i) + ".jpg")
    test_image = tf.image.decode_image(test_image, channels = 1, dtype = tf.float32)
    test_image = np.array(test_image)
    test_image = test_image.reshape(1,
                                    test_image.shape[0],
                                    test_image.shape[1],
                                    test_image.shape[2])
    y_pred = model.predict(test_image)
    print("# ===== Prediction Result {} ====== #".format(i))
    print(y_pred.shape)
    pred_image = np.squeeze(y_pred)
    plt.imshow(pred_image)
    plt.show()