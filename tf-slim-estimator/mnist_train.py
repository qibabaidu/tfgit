import tensorflow as tf

from datasets import cifar10,mnist,flowers
from model import lenet, load_mnist_batch
from model import load_flower_batch
from nets.inception_v3 import inception_v3

slim = tf.contrib.slim

flags = tf.app.flags
flags.DEFINE_string('data_dir', '/tmp/mnist',
                    'Directory with the mnist data.')
flags.DEFINE_integer('batch_size', 16, 'Batch size.')
flags.DEFINE_integer('num_batches', None,
                     'Num of batches to train (epochs).')
flags.DEFINE_string('train_log', './log/train',
                    'Directory with the log data.')
FLAGS = flags.FLAGS



def main(args):
    # load the dataset
    dataset = flowers.get_split('train', FLAGS.data_dir)
    # dataset = cifar10.get_split('train', FLAGS.data_dir)
    # load batch of dataset
    images, image_raw, labels = load_flower_batch(
        dataset,
        FLAGS.batch_size,
        is_training=True)
    #images, labels = load_mnist_batch(
    #    dataset,
    #    FLAGS.batch_size,
    #    is_training=True)
    # run the image through the model
    # predictions = lenet(images)
    logits, end_points = inception_v3(images, dataset.num_classes)

    # get the cross-entropy loss
    one_hot_labels = slim.one_hot_encoding(
        labels,
        dataset.num_classes)
    slim.losses.softmax_cross_entropy(
        logits,
        one_hot_labels)
    total_loss = slim.losses.get_total_loss()
    tf.summary.scalar('loss', total_loss)

    # use RMSProp to optimize
    optimizer = tf.train.RMSPropOptimizer(0.001, 0.9)

    # create train op
    train_op = slim.learning.create_train_op(
        total_loss,
        optimizer,
        summarize_gradients=True)

    # run training
    slim.learning.train(
        train_op,
        FLAGS.train_log,
        save_summaries_secs=30,
        number_of_steps=5000, 
        save_interval_secs=60)


if __name__ == '__main__':
    tf.app.run()
