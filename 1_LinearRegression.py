'''
Lecturer: Sung Kim
회귀분석
How fit the line to our (training) data
:최소자승법 = cost function이 가장 낮은 선 

1. Optimizer를 이용해 Cost 함수 만들고
2. feed_dict을 통해 W, b에 데이터를 주며 학습 한다
'''

import tensorflow as tf

## H(x) = Wx + b ##

## Variable: Tensorflow가 학습하는 과정에서 변형을 하는 변수: Trainable variable
## tf.random_normal: 몇차원 array인가?
W = tf.Variable(tf.random_normal([1]), name = 'weight')
b = tf.Variable(tf.random_normal([1]), name = 'bias')
X = tf.placeholder(tf.float32, shape = [None])
Y = tf.placeholder(tf.float32, shape = [None])

hypothesis = X * W + b

 ## cost(W,b): 최소자승법 식 
cost = tf.reduce_mean(tf.square(hypothesis - Y))

## minimize: 최소 값을 자동으로 반환
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
## train 노드를 실행
train = optimizer.minimize(cost)

## Launch the graph in a session
sess = tf.Session()
## To use W, b, we need to initializes global variables in the graph
sess.run(tf.global_variables_initializer())

## Fit the line
## 2001번 learning 시킬 것이고 20번째마다 출력
for step in range(2001):
    cost_val, W_val, b_val, _ = sess.run([cost, W, b, train], 
                                feed_dict = {X:[1,2,3], Y:[1,2,3]})
    
    if step % 20 == 0:
        print(step, cost_val, W_val, b_val)

