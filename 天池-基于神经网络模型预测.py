#天池-基于神经网络模型预测

#完整神经网络样例程序
import tensorflow as tf
from numpy.random import RandomState

#1. 定义神经网络的参数，输入和输出节点。
batch_size = 8

w1= tf.Variable(tf.random_normal([6, 7], stddev=1, seed=1))
w2= tf.Variable(tf.random_normal([7, 1], stddev=1, seed=1))
x = tf.placeholder(tf.float32,  name="x-input")
y_= tf.placeholder(tf.float32,  name='y-input')

#2. 定义前向传播过程，损失函数及反向传播算法。
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)
#cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0))) 

mse = tf.reduce_sum(tf.square(y_ -  y))
#train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)
train_step = tf.train.AdamOptimizer(0.001).minimize(mse)

# #3. 生成模拟数据集。
# rdm = RandomState(1)
# X = rdm.rand(128,2)
# Y = [[int(x1+x2 < 1)] for (x1, x2) in X]



#3.读取csv至字典x,y
import csv

# 读取csv至字典
csvFile = open(r'G:\0研究生\tianchiCompetition\训练小样本.csv', "r")
reader = csv.reader(csvFile)
#print(reader)

# 建立空字典
result = {}

i=0
for item in reader:
    if reader.line_num==1:
        continue
    result[i]=item
    i=i+1

 # 建立空字典   
j=0
xx={}
yy={}
for i in list(range(29)):
    xx[j]=result[i][1:-1]
    yy[j]=result[i][-1]
    # print(x[j])
    # print(y[j])
    j=j+1

csvFile.close()

##3.1字典转换成list
X=[]
Y=[]
for i in xx.values():
    X.append(i)
    
for j in xx.values():
    X.append(j)    


#4. 创建一个会话来运行TensorFlow程序。
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    # 输出目前（未经训练）的参数取值。
    print("w1:", sess.run(w1))
    print("w2:", sess.run(w2))
    print("\n")

    # 训练模型。
    STEPS = 4
    for i in range(STEPS):
        start = (i*batch_size) % 29
        end = (i*batch_size) % 128 + batch_size
        sess.run(train_step, feed_dict={x: X[start:end], y_: Y[start:end]})
            # total_cross_entropy = sess.run(cross_entropy, feed_dict={x: X, y_: Y})
            # print("After %d training step(s), cross entropy on all data is %g" % (i, total_cross_entropy))
        total_mse=sess.run(mse,feed_dict={x: X, y_: Y})
        print("After %d training step(s), mse on all data is %g" % (i, total_mse))

    # 输出训练后的参数取值。
    print("\n")
    print("w1:", sess.run(w1))
    print("w2:", sess.run(w2))

