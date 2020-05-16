import tensorflow as tf

with tf.variable_scope("Data"):
    # 随机100条样本，每条样本一个特征值，值正态分布
    x = tf.random_normal(shape=[100, 1], mean=5.0, stddev=1.0, name="x_data")
    # 100条样子的目标值。人为设定权重0.8，偏置1.0
    y_true = tf.matmul(x, [[0.8]]) + 1.0

with tf.variable_scope('Model'):
    weight = tf.Variable(tf.random_normal(shape=[1, 1],
                                          mean=0.0,
                                          stddev=1.0,
                                          name="weight"))
    bias = tf.Variable(0.0, name='bias')
    y_predict = tf.matmul(x, weight) + bias

with tf.variable_scope("Loss"):
    loss = tf.reduce_mean(tf.square(y_true - y_predict))

with tf.variable_scope("Optimizer"):
    train = tf.train.GradientDescentOptimizer(0.03).minimize(loss)

var_init = tf.global_variables_initializer()

# 收集需要在board上显示的tensor
tf.summary.scalar(name="Loss", tensor=loss)
tf.summary.histogram(name="weight", values=weight)
summary_merged = tf.summary.merge_all()

with tf.Session() as sess:
    # 建立事件文件
    fw = tf.summary.FileWriter(logdir="summary", graph=sess.graph)

    sess.run(var_init)
    print("初始化 权重：{} 偏置：{}".format(weight.eval(), bias.eval()))

    for i in range(3000):
        # 训练
        sess.run(train)
        if i % 10 == 0:
            # 打印
            print("第{}次优化，权重：{} 偏置：{}".format(i, weight.eval(), bias.eval()))
            # 保存参数变化
            fw.add_summary(summary=sess.run(summary_merged), global_step=i)



