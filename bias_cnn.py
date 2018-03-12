#coding=utf-8
import numpy as np
from scipy import misc # feel free to use another image loader
import tensorflow as tf
import rospy
from geometry_msgs.msg import Twist
from math import radians

actions = [[1,0,0,0,0],[0,1,0,0,0],[0,0,1,0,0],[0,0,0,1,0],[0,0,0,0,1]]


batch_size = 100
num_epochs = 10000
trainable = 1
check_point_dir = "model_saved/model" #新版没有ckpt!!


class Steering():
    def __init__(self,act):
        self.s = act
        rospy.init_node('steering',anonymous=False)
        rospy.on_shutdown(self.shutdown)
        self.cmd_vel = rospy.Publisher('cmd_vel_mux/input/navi', Twist, queue_size=10)
        r = rospy.Rate(5)
        turn_cmd = Twist()
        turn_cmd.linear.x = 0
        turn_cmd.angular.z = radians(30)
        turn_cmd1 = Twist()
        turn_cmd1.linear.x = 0
        turn_cmd1.angular.z = radians(60)
        move_cmd = Twist()
        move_cmd.linear.x = 0
        move_cmd.angular.z = -radians(30)
        move_cmd1 = Twist()
        move_cmd1.linear.x = 0
        move_cmd1.angular.z = -radians(60)
        #s=l.index(1)
        while not rospy.is_shutdown():
            if self.s==0:
                for x in range(0,5):
                    self.cmd_vel.publish(turn_cmd1)
                    r.sleep()
                self.cmd_vel.publish(Twist())
                rospy.spin()
            elif self.s==1:
                for x in range(0,5):
                    self.cmd_vel.publish(turn_cmd)
                    r.sleep()
                self.cmd_vel.publish(Twist())
                rospy.spin()
            elif self.s==2:
                self.cmd_vel.publish(Twist())
                rospy.spin()
            elif self.s==3:
                for x in range(0,5):
                    self.cmd_vel.publish(move_cmd)
                    r.sleep()
                self.cmd_vel.publish(Twist())
                rospy.spin()
            elif self.s==4:
                for x in range(0,5):
                    self.cmd_vel.publish(move_cmd1)
                    r.sleep()
                self.cmd_vel.publish(Twist())
                rospy.spin()
            else:
                self.cmd_vel.publish(Twist())
                rospy.spin()
    def shutdown(self):
        self.cmd_vel.publish(Twist())


def label_to_acition(dir_str = "new_labels.txt"):
    f = open(dir_str)
    label_list = []
    label_num = 0
    while (1):
        line = f.readline()
        if len(line) > 0:
            label_num += 1
            line.replace('\n', '')
            line = float(line)
            if line > -1.3 and line < -0.6:
                label_list.append(actions[0])
            if line >= -0.6 and line < -0.3:
                label_list.append(actions[1])
            if line >= -0.3 and line <= 0.3:
                label_list.append(actions[2])
            if line > 0.3 and line <= 0.6:
                label_list.append(actions[3])
            if line > 0.6 and line < 1.3:
                label_list.append(actions[4])
        else:
            f.close()
            break
    label_list = np.asarray(label_list)
    return label_num,label_list


def get_img(img_num,str = "database"): #img_num=label_num
    images = []
    for i in range(0,img_num):
        str1 = '%s/%d.jpg' %(str,i)
        #list_img.append(str1)
        images.append(misc.imread(str1))

    images = np.asarray(images)  # 统一转化为np的array
    return images


#训练集生成batch
def create_batches(total_size,images,labels,batch_size = 100):
    while (True):
        for i in range(0, total_size, batch_size):
            yield (images[i:i + batch_size], labels[i:i + batch_size])#这是一个生成器，yield返回一个生成器对象（有return功能）
#input:



#imgs = tf.placeholder(tf.float32,shape=[None,60,80,3])
#lbls = tf.placeholder(tf.float32, shape = [None,5])
#keep_prob = tf.placeholder(tf.float32)

#weight,bias and cnn layer


def weight_variable(shape,name):
    initial = tf.truncated_normal(shape,stddev = 0.01)
    return tf.Variable(initial,name)


def bias_variable(shape,name):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial , name)


def conv2d(x,W,stride):
    return tf.nn.conv2d(x , W , strides=[1,stride,stride,1] , padding = "SAME")


def max_pool_2x2(x):
    return tf.nn.max_pool(x , ksize=[1,2,2,1] , strides = [1,2,2,1] , padding = "SAME")


def creat_net():
    #input
    imgs = tf.placeholder(tf.float32,shape=[None,60,80,3])
    keep_prob = tf.placeholder(tf.float32)
    # conv network weight
    W_conv1 = weight_variable([8,8,3,32],"W_conv1") #卷积核边长为5，输入层深度为3，输出层深度为32
    b_conv1 = bias_variable([32],"b_conv1") #偏置

    W_conv2 = weight_variable([4,4,32,64],"W_conv2") #与上层深度应一致，池化无法改变深度
    b_conv2 = bias_variable([64],"b_conv2")

    W_conv3 = weight_variable([2,2,64,64],"W_conv3")
    b_conv3 = bias_variable([64],"b_conv3")

    #conv network
    h_conv1 = tf.nn.relu(conv2d(imgs,W_conv1,2) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2,2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3,2) + b_conv3)
    h_pool3 = max_pool_2x2(h_conv3)

    #将张量抽成一维向量
    pool_shape = h_pool3.get_shape().as_list()
    #注意这里的pool_shape[0]是一个batch中的数量
    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
    h_pool3_flat = tf.reshape(h_pool3,[-1,nodes])

    #full connected network
    W_fc1 = weight_variable([nodes,256],"W_fc1")
    b_fc1 = bias_variable([256],"b_fc1")

    W_fc2 = weight_variable([256,5],"W_fc2")
    b_fc2 = bias_variable([5],"b_fc2")

    h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)
    h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)

    pre_Q = tf.matmul(h_fc1_drop,W_fc2) + b_fc2
    pre_action = tf.nn.softmax(pre_Q)
    return imgs,keep_prob,pre_Q,pre_action


def train_net(imgs,keep_prob,pre_Q,total_size,images,labels,trainable = 1):
    #input label
    lbls = tf.placeholder(tf.float32,shape=[None,5])
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=lbls,logits=pre_Q)
    loss = tf.reduce_mean(cross_entropy)#cross entropy is an array
    train_step = tf.train.AdamOptimizer(1e-5).minimize(loss)
    batch_generator = create_batches(total_size,images,labels)
    #record
    tf.summary.scalar('loss',loss)
    merged = tf.summary.merge_all()
    #saver
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        train_writer = tf.summary.FileWriter("logs/train", sess.graph)
        if trainable:
            for i in range(num_epochs):
                img_in,label_in = batch_generator.next()
                _,train_loss = sess.run([train_step,loss],feed_dict = {imgs:img_in,lbls:label_in,keep_prob:0.5})
                if i%100 == 0:
                    print "train_loss is %f" % train_loss
                    train_loss_record = sess.run(merged, feed_dict={imgs: img_in, lbls: label_in, keep_prob: 1})
                    train_writer.add_summary(train_loss_record, i)
                    saver.save(sess,check_point_dir)
        else:
            saver.restore(sess, './model_saved/model')
            list_use = []
            img_use = misc.imread('/home/z/PycharmProjects/RL/database/2000.jpg')
            list_use.append(img_use)
            list_use = np.asarray(list_use)
            tf.reshape(list_use,[1,60,80,3])
            action = sess.run(pre_action,feed_dict={imgs:list_use,keep_prob:1})
            print action
            act = np.argmax(action)
            print act
        return act

if __name__ == "__main__":
    label_num,labels = label_to_acition()
    img_in = get_img(label_num)
    imgs, keep_prob, pre_Q, pre_action = creat_net()
    act = train_net(imgs,keep_prob,pre_Q,label_num,img_in,labels,0)
    print act
    try:
        Steering(act)
    except:
        rospy.loginfo("node terminated.")


