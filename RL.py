#coding=utf-8
import cv2
import numpy as np
tracker = cv2.TrackerKCF_create()

cap = cv2.VideoCapture(1)
width = 640
flag = 1
num = 7157
txt_name = "new_labels_7157+.txt"
f = open(txt_name,'a')  #追加写入

def normlize(x,width):
    width_half = width/2
    norm_val = float((x-width_half))/float(width_half)
    return norm_val

while(1):
    if flag==1:
        flag+=1
        ok,frame = cap.read()
        bbox = cv2.selectROI(frame,False)
        ok = tracker.init(frame,bbox)

    else:
        ok,frame = cap.read()
        img = frame.copy()
        ok,bbox = tracker.update(frame)
        if ok:
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            p3 = (int(bbox[0] + bbox[2]/2) , int(bbox[1] + bbox[3]/2))
            cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
            cv2.circle(frame,p3,3,(0,0,255))
            norm_val = normlize(p3[0],width)

            save_img = cv2.resize(img,(80,60))
            #save_img = img
            #print img.shape
            img_name = "/home/z/PycharmProjects/RL/database/%d.jpg"%num
            cv2.imwrite(img_name,save_img)

            str = "%.1f\n" % norm_val    #比例问题
            print str
            f.write(str)

            num+=1


        else:
            cv2.putText(frame,"tracking failure",(100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
        cv2.imshow("tracking" , frame)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

