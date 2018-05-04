'''
Lecturer: Sung Kim

여기서는 Cost Function을 조금 더 깊게 살펴본다
- Cost Function 설계할 때 산꼭대기 경사가 하나(Convex Function)인지 확인!!
'''
import tensorflow as tf
import matplotlib.pyplot as plt
x_data = [1,2,3]
y_data = [1,2,3]

W = tf.Variable(tf.random_normal([1]), name = 'weight')
X = tf.placeholder(tf.float32, shape = [None])
Y = tf.placeholder(tf.float32, shape = [None])
# Our hypothesis for Linear model W*X
hypothesis = W*X

## 예측값 - 실제값
cost = tf.reduce_mean(tf.square(hypothesis - Y))

## Cost를 minimize하는 함수를 직접 만듬
learning_rate = 0.1
gradient = tf.reduce_mean((W*X-Y)*X)
descent = W - learning_rate * gradient
update = W.assign(descent)
## optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.1)
## train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(21):
    sess.run(update, feed_dict={X: x_data, Y: y_data})
    print(step, sess.run(cost, feed_dict={X: x_data, Y:y_data}), sess.run(W))
    ## cost가 점점 낮아지는 것을 확인할 수 있다 