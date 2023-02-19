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

DATASET_DIR = pathlib.Path("hyper-kvasir/segmented-images")
MASK_DATA = DATASET_DIR.joinpath("masks")
CLASS_DATA = DATASET_DIR.joinpath("images")
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

training_data = []
labels = []
CLASS_DATA = CLASS_DATA.joinpath("polyps")
for file in os.listdir(CLASS_DATA):
    training_data.append(skimage.transform.resize(skimage.io.imread(CLASS_DATA.joinpath(file), as_gray=True), (256, 256, 1)))
    labels.append(1)

mask = []
for file in os.listdir(MASK_DATA):
    mask.append(skimage.transform.resize(skimage.io.imread(MASK_DATA.joinpath(file), as_gray=True), (256, 256, 1)))


train = training_data[(len(training_data)//100) * 20:]
test = training_data[:(len(training_data)//100) * 20]

train_labels = labels[(len(labels)//100) * 20:]
test_labels = labels[:(len(labels)//100) * 20]

train_mask = mask[(len(mask)//100) * 20:]
test_mask = mask[:(len(labels)//100) * 20]

#train = imread_collection(os.listdir(CLASS_DATA.joinpath("polyps")))

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

history = model.fit({'input': np.array(train)},
                    {'classification': np.array(train_labels), 'segmentation': np.array(train_mask)},
                    epochs=50, batch_size=BATCH_SIZE,
                    verbose=1,
                    validation_data=({'input': np.array(test)},
                    {'classification': np.array(test_labels), 'segmentation': np.array(test_mask)}),
                    callbacks=[cp_callback]
                    )

# Train the model with the new callback
#model.fit(train_images,
#          train_labels,
#          epochs=10,
#          validation_data=(test_images, test_labels),
#          callbacks=[cp_callback])  # Pass callback to training

