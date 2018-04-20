'''
Lecturer: Sung Kim

Deep Learning
Unsupervised
- unlabeled data (No training data set)
Supervised
- Learning with labeled examples
- Regression / Binary Classification / Multi-label Classification 

Tensorflow is an open source software library for numerical computation using data flow graphs. Also, can use python

Data Flow Graph
- Nodes in the graph represent mathematical operations
- Edges represent the multidimensional data arrays(=Tensor) communicated between them

TensorFlow Mechanics
1. Build graph using TensorFlow operations
2. Feed data and run graph (operation)
3. Update variables in the graph (and return values)
'''

import tensorflow as tf

## Let's make 'add' computational graph
## Create a constant op which is added as a node to the default graph
node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0, tf.float32)
node3 = tf.add(node1, node2)

## Make TF session
sess = tf.Session()
## Run TF     
print(sess.run(node1, node2, node3))

## Feed the data !
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b

print(sess.run(adder_node, feed_dict={a:3, b:4.5}))
print(sess.run(adder_node, feed_dict={a:[1,3], b:[2,4]}))

