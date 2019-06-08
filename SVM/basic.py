import tensorflow as tf 
import numpy as np 

a=tf.constant([[1,2,3],[4,5,6]])
at=tf.reshape(a, [-1,1])

sess=tf.Session()

result_a=sess.run(a)
result_at=sess.run(at)

print result_a
print result_at
