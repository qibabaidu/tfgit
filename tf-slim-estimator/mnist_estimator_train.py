import tensorflow as tf

from datasets import cifar10,mnist,flowers
from model import lenet, load_mnist_batch
from model import load_flower_batch
from nets.inception_v3 import inception_v3
import time
import os

slim = tf.contrib.slim

flags = tf.app.flags
flags.DEFINE_string('data_dir', '/tmp/mnist', 'Directory with the mnist data.')
flags.DEFINE_string('train_dir', './log/train', 'Directory with the log data.')
flags.DEFINE_integer('num_classes', 5, 'Number of classes to train')
FLAGS = flags.FLAGS

train_params = {'data_dir': FLAGS.data_dir,
                'split_name': 'train',
                'is_training' : True,
                'mode': tf.estimator.ModeKeys.TRAIN,
                'num_classes' : FLAGS.num_classes,
                'threads': 16,
                'shuffle_buff': 100000,
                'batch_size': 32,
                'steps' : 5000}

eval_params  = {'data_dir': FLAGS.data_dir,
                'split_name': 'validation',
                'is_training' : False,
                'mode': tf.estimator.ModeKeys.EVAL,
                'num_classes' : FLAGS.num_classes,
                'threads': 8,
                'batch_size': 8,
                'eval_steps' : 100}

def dataset_input_fn(params):
    def _input_fn():
        dataset = flowers.get_split(params['split_name'], params['data_dir'])
        images, image_raw, labels = load_flower_batch(dataset,
                                                      params['batch_size'],
                                                      is_training=params['is_training'])
        return images, labels
    return _input_fn

def model_fn(features, labels, mode, params):

    logits, end_points = inception_v3(features, FLAGS.num_classes)

    if mode in (tf.estimator.ModeKeys.PREDICT, tf.estimator.ModeKeys.EVAL):
        predicted_indices = tf.argmax(input=logits, axis=1)
        probabilities = end_points['Predictions']

    if mode in (tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL):
        global_step = tf.train.get_or_create_global_step()
        label_indices = tf.argmax(input=labels, axis=1)
        loss = tf.losses.softmax_cross_entropy(onehot_labels=tf.one_hot(labels, FLAGS.num_classes),
                                               logits=logits)
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = { 'classes' : predicted_indices, 'probabilities' : probabilities }
        export_outputs = { 'predictions': tf.estimator.export.PredictOutput(predictions) }
        return tf.estimator.EstimatorSpec( mode, predictions=predictions, export_outputs=export_outputs)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.RMSPropOptimizer(0.001, 0.9)
        train_op = optimizer.minimize(loss, global_step=global_step)
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
    if mode == tf.estimator.ModeKeys.EVAL:
        eval_metric_ops = {'accuracy': tf.metrics.accuracy(label_indices, predicted_indices)}
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=eval_metric_ops)

distribution = tf.contrib.distribute.MirroredStrategy(num_gpus=1)
run_config = tf.estimator.RunConfig(
    save_checkpoints_secs=300,
    keep_checkpoint_max=5,
    model_dir=FLAGS.train_dir,
    session_config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True),
    train_distribute=distribution)

estimator = tf.estimator.Estimator(model_fn=model_fn, config=run_config)

train_spec = tf.estimator.TrainSpec(input_fn=dataset_input_fn(train_params),
                                    max_steps=train_params['steps'])
eval_spec  = tf.estimator.EvalSpec(input_fn=dataset_input_fn(eval_params),
                                   steps=eval_params['eval_steps'])

# 要让input_fn可以call
estimator.train(input_fn=lambda : dataset_input_fn(train_params), max_steps=5000)

estimator.evaluate(input_fn=lambda: dataset_input_fn(eval_params), steps=10)