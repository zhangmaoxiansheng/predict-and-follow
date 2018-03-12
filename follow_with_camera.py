#coding=utf-8
from bias_cnn_follower_v2 import Steering
from bias_cnn_follower_v2 import bias_cnn
import cv2

if __name__ == "__main__":
    pre_cnn = bias_cnn()
    cap = cv2.VideoCapture(0)#若用外置摄像头应为1
    pre_cnn.restore_net()
    while(1):
        ok,img = cap.read()
        img = cv2.resize(img,(80,60))
        act = pre_cnn.use_net_with_cam(img)
        Steering(act)
        #ros 需不需要暂停
        cv2.imshow("collect", img)
        if cv2.waitKey(250) & 0xff == ord('q'):
            pre_cnn.sess.close()
            break

    pre_cnn.sess.close()
    cap.release()
    cv2.destroyAllWindows()