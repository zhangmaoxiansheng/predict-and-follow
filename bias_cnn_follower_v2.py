#coding=utf-8
import numpy as np
from scipy import misc # feel free to use another image loader
import tensorflow as tf
import rospy
from geometry_msgs.msg import Twist
from math import radians

class Steering:
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


class bias_cnn:
    def __init__(self,

                 num_batch=100,
                 eposide=10000,
                 img_dir="database",
                 label_dir = "labels.txt",
                 save_path = "model_saved/model",
                 test_path = "database"):
        self.actions = [[1,0,0,0,0],[0,1,0,0,0],[0,0,1,0,0],[0,0,0,1,0],[0,0,0,0,1]]
        self.num_batch = num_batch
        self.epo = eposide
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.save_path = save_path
        self.test_path = test_path

        self.build_net()#初始化仅构建网络
        self.saver = tf.train.Saver()#先构建网络才能save

    def label_to_acition(self):
        self.labels = []
        f = open(self.label_dir)
        self.total_num = 0
        while (1):
            line = f.readline()
            if len(line) > 0:
                self.total_num += 1
                line.replace('\n', '')
                line = float(line)
                if line > -1.3 and line < -0.6:
                    self.labels.append(self.actions[0])
                if line >= -0.6 and line < -0.3:
                    self.labels.append(self.actions[1])
                if line >= -0.3 and line <= 0.3:
                    self.labels.append(self.actions[2])
                if line > 0.3 and line <= 0.6:
                    self.labels.append(self.actions[3])
                if line > 0.6 and line < 1.3:
                    self.labels.append(self.actions[4])
            else:
                f.close()
                break
        self.labels = np.asarray(self.labels)


    def get_img(self):
        self.images = []
        for i in range(0, self.total_num):
            str1 = '%s/%d.jpg' % (self.img_dir, i)
            # list_img.append(str1)
            self.images.append(misc.imread(str1))

        self.images = np.asarray(self.images)  # 统一转化为np的array

    def create_batch(self):
        while (True):
            for i in range(0, self.total_num, self.num_batch):
                yield (self.images[i:i + self.num_batch], self.labels[i:i + self.num_batch])  # 这是一个生成器，yield返回一个生成器对象（有return功能）

    @staticmethod
    def weight_variable(shape, name):
        initial = tf.truncated_normal(shape, stddev=0.01)
        return tf.Variable(initial, name)

    @staticmethod
    def bias_variable(shape,name):
        initial = tf.constant(0.1,shape=shape)
        return tf.Variable(initial,name)

    @staticmethod
    def conv2d(x,W,stride):
        return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding="SAME")

    @staticmethod
    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    def build_net(self):
        #input
        self.imgs = tf.placeholder(tf.float32, shape=[None, 60, 80, 3])
        self.lbls = tf.placeholder(tf.float32, shape=[None, 5])
        self.keep_prob = tf.placeholder(tf.float32)
        #cnn
        W_conv1 = self.weight_variable([8, 8, 3, 32], "W_conv1")  # 卷积核边长为5，输入层深度为3，输出层深度为32
        b_conv1 = self.bias_variable([32], "b_conv1")  # 偏置

        W_conv2 = self.weight_variable([4, 4, 32, 64], "W_conv2")  # 与上层深度应一致，池化无法改变深度
        b_conv2 = self.bias_variable([64], "b_conv2")

        W_conv3 = self.weight_variable([2, 2, 64, 64], "W_conv3")
        b_conv3 = self.bias_variable([64], "b_conv3")

        # conv network
        h_conv1 = tf.nn.relu(self.conv2d(self.imgs, W_conv1, 2) + b_conv1)
        h_pool1 = self.max_pool_2x2(h_conv1)

        h_conv2 = tf.nn.relu(self.conv2d(h_pool1, W_conv2, 2) + b_conv2)
        h_pool2 = self.max_pool_2x2(h_conv2)

        h_conv3 = tf.nn.relu(self.conv2d(h_pool2, W_conv3, 2) + b_conv3)
        h_pool3 = self.max_pool_2x2(h_conv3)

        # 将张量抽成一维向量
        pool_shape = h_pool3.get_shape().as_list()
        # 注意这里的pool_shape[0]是一个batch中的数量
        nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
        h_pool3_flat = tf.reshape(h_pool3, [-1, nodes])

        # full connected network
        W_fc1 = self.weight_variable([nodes, 256], "W_fc1")
        b_fc1 = self.bias_variable([256], "b_fc1")

        W_fc2 = self.weight_variable([256, 5], "W_fc2")
        b_fc2 = self.bias_variable([5], "b_fc2")

        h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)
        h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)

        self.pre_Q = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
        self.pre_action = tf.nn.softmax(self.pre_Q)

    def train_net(self):
        self.label_to_acition()
        self.get_img()

        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.lbls, logits=self.pre_Q)
        self.loss = tf.reduce_mean(cross_entropy)
        train_step = tf.train.AdamOptimizer(1e-5).minimize(self.loss)
        batch_generator = self.create_batch()
        # record
        tf.summary.scalar('loss', self.loss)
        merged = tf.summary.merge_all()
        # saver
        #self.saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            train_writer = tf.summary.FileWriter("logs/train", sess.graph)

            for i in range(self.epo):
                img_in, label_in = batch_generator.next()
                _, train_loss = sess.run([train_step, self.loss],
                                             feed_dict={self.imgs: img_in, self.lbls: label_in, self.keep_prob: 0.5})
                if i % 100 == 0:
                    print "train_loss is %f" % train_loss
                    train_loss_record = sess.run(merged, feed_dict={self.imgs: img_in, self.lbls: label_in, self.keep_prob: 1})
                    train_writer.add_summary(train_loss_record, i)
                    self.saver.save(sess, self.save_path)

    def use_net(self,test_num):
        with tf.Session() as sess:
            self.saver.restore(sess,self.save_path)
            list_use = []
            str2 = '%s/%d.jpg' %(self.test_path,test_num)
            image_use = misc.imread(str2)#测试路径
            list_use.append(image_use)
            list_use = np.asarray(list_use)
            tf.reshape(list_use, [1, 60, 80, 3])  # 将图片reshape
            action = sess.run(self.pre_action, feed_dict={self.imgs: list_use, self.keep_prob: 1})
            #print action
            act = np.argmax(action)
            #print act
        return act

    def restore_net(self):
        self.sess = tf.Session()
        self.saver.restore(self.sess, self.save_path)


    def use_net_with_cam(self,img):
        #需要先加载网络以及session用restore net即可
        list_use = []
        list_use.append(img)
        list_use = np.asarray(list_use)
        tf.reshape(list_use, [1,60,80,3])
        action = self.sess.run(self.pre_action, feed_dict={self.imgs: list_use, self.keep_prob: 1})
        # print action
        act = np.argmax(action)
        return act



if __name__ == "__main__":
    pre_cnn = bias_cnn()
    act = pre_cnn.use_net(2000)
    #print act
    try:
        Steering(act)
    except:
        rospy.loginfo("node terminated.")