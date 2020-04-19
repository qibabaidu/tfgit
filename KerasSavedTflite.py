"""
TF 2.0
"""

import tensorflow as tf
import tempfile
import zipfile
import os
import sys
if sys.version_info.major >= 3:
    import pathlib
else:
    import pathlib2 as pathlib

# Add `models` to the python path.
models_path = os.path.join(os.getcwd(), "models")
sys.path.append(models_path)

saved_model_dir = "./models/pb"

batch_size = 128
num_classes = 10
epochs = 10

# input image dimensions
img_rows, img_cols = 28, 28

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# NCHW OR NHWC
if tf.keras.backend.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test  = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

layer = tf.keras.layers

model = tf.keras.Sequential([   layer.Conv2D(32, 5, padding='same', activation='relu', input_shape=input_shape),
                                layer.MaxPooling2D((2, 2), (2, 2), padding="same"),
                                layer.BatchNormalization(),
                                layer.Conv2D(64, 5, padding='same', activation='relu'),
                                layer.MaxPooling2D((2, 2), (2, 2), padding='same'),
                                layer.Flatten(),
                                layer.Dense(1024, activation="relu"),
                                layer.Dropout(0.4),
                                layer.Dense(num_classes, activation="softmax")
                            ])
model.summary()

logdir = tempfile.mkdtemp()
print('Writing training logs to ' + logdir)

callbacks = [tf.keras.callbacks.TensorBoard(log_dir=logdir, profile_batch=0)]

model.compile( loss = tf.keras.losses.categorical_crossentropy,
               optimizer = 'adam',
               metrics = ['accuracy'] )

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          callbacks=callbacks,
          validation_data=(x_test, y_test))
tf.keras.experimental.export_saved_model(model, saved_model_dir)

# tf lite
"""
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
tflite_model = converter.convert()
tflite_models_dir = pathlib.Path("./models/mnist_tflite_models/")
tflite_models_dir.mkdir(exist_ok=True, parents=True)
tflite_model_file = tflite_models_dir/"mnist_model.tflite"
tflite_model_file.write_bytes(tflite_model)
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
tflite_quant_model = converter.convert()
tflite_model_quant_file = tflite_models_dir/"mnist_model_quant.tflite"
tflite_model_quant_file.write_bytes(tflite_quant_model)
"""

infer_model = tf.keras.experimental.load_from_saved_model(saved_model_dir)

model.compile( loss = tf.keras.losses.categorical_crossentropy,
               optimizer = 'adam',
               metrics = ['accuracy'] )

_,acc = model.evaluate(x_test, y_test)
print("Restore model, accuracy: {:5.2f}%".format(100*acc))
