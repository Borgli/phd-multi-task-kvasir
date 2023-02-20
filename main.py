import pathlib

import skimage.io

import os
import skimage.transform as trans
import numpy as np
import tensorflow as tf

from model import create_model

BATCH_SIZE = 32
IMG_HEIGHT = 256
IMG_WIDTH = 256

DATASET_DIR = pathlib.Path("data")
TEST_DATA = DATASET_DIR.joinpath("test")
TRAIN_DATA = DATASET_DIR.joinpath("train")
VALIDATION_DATA = DATASET_DIR.joinpath("validation")

'''
train_ds = tf.keras.utils.image_dataset_from_directory(
    CLASS_DATA,
    labels=None,
    color_mode="grayscale",
    #validation_split=0.2,
    #subset="both",
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE
)

mask_ds = tf.keras.utils.image_dataset_from_directory(
    MASK_DATA,
    labels=None,
    color_mode="grayscale",
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE
)
'''
model = create_model()

'''
train_ds = train_ds.unbatch()
mask_ds = mask_ds.unbatch()

labels = list(map(lambda x: 1, train_ds))

train_ds = list(train_ds.as_numpy_iterator())
mask_ds = list(mask_ds.as_numpy_iterator())
'''


def read_images(path, get_labels=False):
    images = []
    labels = []
    for file in os.listdir(path):
        images.append(skimage.transform.resize(skimage.io.imread(path.joinpath(file),
                                                                 as_gray=True), (256, 256, 1)))
        if get_labels:
            labels.append(1)

    if get_labels:
        return images, labels
    else:
        return images


train_images, train_labels = read_images(TRAIN_DATA.joinpath("images"), get_labels=True)
train_masks = read_images(TRAIN_DATA.joinpath("masks"))
test_images, test_labels = read_images(TEST_DATA.joinpath("images"), get_labels=True)
test_masks = read_images(TEST_DATA.joinpath("masks"))

# train = imread_collection(os.listdir(CLASS_DATA.joinpath("polyps")))

'''
try:
    history = model.fit({'input': np.array(train)},
                        {'classification': np.array(labels), 'segmentation': np.array(mask)},
                        epochs=50, batch_size=BATCH_SIZE,
                        verbose=1
                        )

except KeyboardInterrupt:
    pass
'''

checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

history = model.fit({'input': np.array(train_images)},
                    {'classification': np.array(train_labels), 'segmentation': np.array(train_masks)},
                    epochs=50, batch_size=BATCH_SIZE,
                    verbose=1,
                    validation_data=({'input': np.array(test_images)},
                                     {'classification': np.array(test_labels), 'segmentation': np.array(test_masks)}),
                    callbacks=[cp_callback]
                    )

# Train the model with the new callback
# model.fit(train_images,
#          train_labels,
#          epochs=10,
#          validation_data=(test_images, test_labels),
#          callbacks=[cp_callback])  # Pass callback to training
