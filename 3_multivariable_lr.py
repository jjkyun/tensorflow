'''
lecturer: Sung Kim

Multi-Variable linear regression
- H(x1, x2, x3) = w1*x1 + w2*x2 + w3*x3 + b

위 가설을 토대로 cost function 정의:: 최소자승법 
cost(W, b) = 1/m * Sum(H(x1, x2, x3) - y)^2

But, what if variable is too many? USE MATRIX
위 가설의 w1*x1 + w2*x2 + w3*x3 는 매트릭스의 곱셈으로 표현된다
-> H(X) = X*W 
'''

import tensorflow as tf

## Actual data 
'''
x1_data = [73., 93., 89., 96., 73.]
x2_data = [80., 88., 91., 98., 66.]
x3_data = [75., 93., 90., 100., 70.]
y_data = [152., 185., 180., 196., 142.]
'''
## 위의 데이터들을 matrix로 표현해보자
x_data = [[73., 80., 75.,], [93., 88., 93.], [89., 91., 90.],
            [96., 98., 100.],[73., 66., 70.]]
y_data = [[152.], [185.], [180.], [196.], [142.]]


## placeholders for a tensor that will be always fed 
'''
x1 = tf.placeholder(tf.float32)
x2 = tf.placeholder(tf.float32)
x3 = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)
'''
X = tf.placeholder(tf.float32, shape = [None, 3]) ## None은 instance가 n개일 수 있다는 뜻
Y = tf.placeholder(tf.float32, shape = [None, 1])

'''
w1 = tf.Variable(tf.random_normal([1]), name='weight1')
w2 = tf.Variable(tf.random_normal([1]), name='weight2')
w3 = tf.Variable(tf.random_normal([1]), name='weight3')
b = tf.Variable(tf.random_normal([1]), name='bias')
'''
W = tf.Variable(tf.random_normal([3, 1]), name = 'weight')
b = tf.Variable(tf.random_normal([1]), name = 'bias')

'''
hypothesis = x1*w1 + x2*w2 + x3*w3 + b
'''
hypothesis = tf.matmul(X, W) + b

## Defining cost function
cost = tf.reduce_mean(tf.square(hypothesis - Y))
## Defining optimizer based on Gradient Descent method
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

## Launch the graph in a session
sess = tf.Session()
## Initalizes global variables in the group
sess.run(tf.global_variables_initializer())
for step in range(2001):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train],
                                    feed_dict = {X: x_data, Y: y_data})
    if step % 10 == 0:
        print(step, "Cost: ", cost_val, "\nPrediction:\n", hy_val)


