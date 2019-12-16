# coding=utf-8
from itertools import count
from collections import defaultdict
from scipy.sparse import csr
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.feature_extraction import DictVectorizer

cols = ['user', 'item', 'rating', 'timestamp']

train=pd.read_csv("./dataset/ml-100k/ua.base", delimiter='\t', names=cols)
test =pd.read_csv("./dataset/ml-100k/ua.test", delimiter='\t', names=cols)
train = train.drop(['timestamp'], axis=1)
test = test.drop(['timestamp'], axis=1)

# DictVectorizer会把数字识别为连续特征，这里把user id和item id强制转为 catogorical identifier
train['item'] = train['item'].apply(lambda x: "c" + str(x))
train["user"] = train["user"].apply(lambda x: "u"+str(x))

test["item"] = test["item"].apply(lambda x: "c"+str(x))
test["user"] = test["user"].apply(lambda x: "u"+str(x))

all_df = pd.concat([train, test])
print("all_df head", all_df.head())

vec = DictVectorizer()
vec.fit_transform(all_df.to_dict(orient='record'))
# 合并训练集与验证集，是为了one hot，用完可以释放掉
del all_df

x_train = vec.transform(train.to_dict(orient='record')).toarray()
x_test = vec.transform(test.to_dict(orient='record')).toarray()

print("x_train shape", x_train.shape)
print("x_test shape", x_test.shape)

y_train = train['rating'].values.reshape(-1, 1)
y_test = test['rating'].values.reshape(-1, 1)
print("y_train shape", y_train.shape)
print("y_test shape", y_test.shape)

n, p = x_train.shape
k = 40
x = tf.placeholder('float', [None, p])
y = tf.placeholder('float', [None, 1])
w0 = tf.Variable(tf.zeros([1]))

w = tf.Variable(initial_value=tf.random_normal(shape=[p], mean=0, stddev=0.1))
v = tf.Variable(tf.random_normal([k, p], mean=0, stddev=0.01))

linear_terms = tf.add(w0, tf.reduce_sum(tf.multiply(w, x), 1, keep_dims=True))
pair_interactions = 0.5 * tf.reduce_sum(tf.subtract(
                    tf.pow(tf.matmul(x, tf.transpose(v)), 2),
                    tf.matmul(tf.pow(x, 2), tf.transpose(tf.pow(v, 2)))), axis=1, keep_dims=True)

y_hat = tf.add(linear_terms, pair_interactions)

lambda_w = tf.constant(0.001, name='lambda_w')
lambda_v = tf.constant(0.001, name='lambda_v')

#  加l2 正则，提高解的泛化能力
l2_norm = 0.5*tf.reduce_sum(
    tf.add(
        tf.multiply(lambda_w, tf.pow(w, 2)),
        tf.multiply(lambda_v, tf.pow(v, 2))
    )
)

error = tf.reduce_mean(tf.square(y-y_hat))
loss = tf.add(error, l2_norm)

train_op = tf.train.GradientDescentOptimizer(learning_rate=0.02).minimize(loss)

epochs = 10

# Launch the graph
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)

    for epoch in range(epochs):
        # iterate over batches
        _, t = sess.run([train_op, loss], feed_dict={x: x_train, y: y_train})
        validate_loss = sess.run(error, feed_dict={x: x_test, y: y_test})
        print("epoch:%d train loss:%f validate_loss:%s" %
              (epoch, t, validate_loss))

    loss = sess.run(loss, feed_dict={x: x_test, y: y_test})
    print("loss:", loss)
    error = sess.run(error, feed_dict={x: x_test, y: y_test})
    RMSE = np.sqrt(error)
    print("rmse:", RMSE)

    predict = sess.run(y_hat, feed_dict={x: x_test[0:10]})
    print("predict:", predict)
