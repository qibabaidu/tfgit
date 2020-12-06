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

file_pattern = f"/home/ajay/tensorflow_datasets/cifar10/1.0.2/cifar10-train.tfrecord*"
files = tf.data.Dataset.list_files(file_pattern)
train_dataset = files.interleave(tf.data.TFRecordDataset,
                                 cycle_length=4,        # 同时处理的数据个数
                                 block_length=2,        # 每次取多少个元素出来
                                 num_parallel_calls=tf.data.experimental.AUTOTUNE)

'''
t1 = t.shuffle(10).batch(2)
#这个是先打乱t的顺序，然后batch
t2 = t.batch(2).shuffle(10)
#这个是打乱batch的顺序
t3 = t.batch(2).repeat(2)
#重复batch，而不是数据
t4 = t.repeat(2).batch(2)
#重复数据，再batch
'''
def read_tfrecord(serialized_example):
    feature_description = {
        'image' : tf.io.FixedLenFeature((), tf.string, ""),
        'label' : tf.io.FixedLenFeature((), tf.int64, -1),
    }
    example = tf.io.parse_single_example(serialized_example, feature_description)
    image = tf.io.decode_jpeg(example['image'], channels=3)

    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.resize(image, (224, 224))
    return image, example['label']

def input_fn(train_dataset):
    cores = multiprocessing.cpu_count()
    train_dataset = train_dataset.map(read_tfrecord, num_parallel_calls=cores)
    train_dataset = train_dataset.shuffle(100)
    # repeat(2)了之后训练时间*2,但是精度确实有提升
    train_dataset = train_dataset.batch(32).repeat(2)
    train_dataset = train_dataset.prefetch(buffer_size=2)
    return train_dataset

model = create_model()
model.fit(input_fn(train_dataset), epochs=5)
