import multiprocessing
from os import getcwd

import tensorflow as tf
from tensorflow.keras import datasets

def create_model():
    input_layer = tf.keras.layers.Input(shape=(224,224,3))
    base_model = tf.keras.applications.MobileNetV2(input_tensor=input_layer,
                                                   weights="imagenet",
                                                   include_top=False)
    # train new last layer
    base_model.trainable = False
    x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    x = tf.keras.layers.Dense(10, activation='softmax')(x)

    model = tf.keras.models.Model(inputs=input_layer, outputs=x)
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['acc'])
    return model

def prepare_cifar10_features_and_labels(features, labels):
    features = tf.cast(features, tf.float32) / 255.0
    features = tf.image.resize(features, (224, 224))
    labels = tf.cast(labels, tf.int64)
    return features, labels

def cifar10_dataset():
    (x_train, y_train), _ = datasets.cifar10.load_data()
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.map(prepare_cifar10_features_and_labels)
    return train_dataset

train_dataset = cifar10_dataset().batch(32)

model = create_model()
model.fit(train_dataset, epochs=5)
