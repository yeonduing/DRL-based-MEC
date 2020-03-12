import cv2
import io
import socket
import struct
import time
import pickle
import zlib
import time
import numpy as np
import math
import queue
import threading
q_max = 50000
nodeBeforeBuff = queue.Queue(q_max)
nodeAfterBuff = queue.Queue(q_max)

nodebeforebufftime = queue.Queue(q_max)
nodeafterbufftime = queue.Queue(q_max)

width = 1920
height = 1080
K = np.array([[width, 0, width / 2], [0, height, height / 2], [0, 0, 1]])

isend=0

cam = cv2.VideoCapture('jk_l.mp4')
cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

f = open("node_only_latency_1920.txt", "w")

def img_mod():
    while True:
        img = nodeBeforeBuff.get()
        # print("hi\n")
        h_, w_ = img.shape[:2]
        # pixel coordinates
        y_i, x_i = np.indices((h_, w_))
        X = np.stack([x_i, y_i, np.ones_like(x_i)], axis=-1).reshape(h_ * w_, 3)  # to homog
        Kinv = np.linalg.inv(K)
        X = Kinv.dot(X.T).T  # normalized coords
        # calculate cylindrical coords (sin\theta, h, cos\theta)
        A = np.stack([np.sin(X[:, 0]), X[:, 1], np.cos(X[:, 0])], axis=-1).reshape(w_ * h_, 3)
        B = K.dot(A.T).T  # project back to image-pixels plane
        # back from homog coords
        B = B[:, :-1] / B[:, [-1]]
        # make sure warp coords only within image bounds
        B[(B[:, 0] < 0) | (B[:, 0] >= w_) | (B[:, 1] < 0) | (B[:, 1] >= h_)] = -1
        B = B.reshape(h_, w_, -1)

        img_rgba = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)  # for transparent borders...
        # warp the image according to cylindrical coords
        img = cv2.remap(img_rgba, B[:, :, 0].astype(np.float32), B[:, :, 1].astype(np.float32), cv2.INTER_AREA,
                         borderMode=cv2.BORDER_TRANSPARENT)
        check = time.time() - nodebeforebufftime.get()
        nodeafterbufftime.put(check)
        f.write(str(check)+'\n')
        if isend==1:
            f.close()
        nodeAfterBuff.put(img)


def frame_read():
    while (cam.isOpened()):
        # print("hi2\n")
        ret, frame = cam.read()

        if frame is None:
            isend=1
            break;
        nodeBeforeBuff.put(frame)
        nodebeforebufftime.put(time.time())



if __name__ == '__main__':
    thread_read_frame = threading.Thread(target=frame_read, daemon=True)
    thread_img_process = threading.Thread(target=img_mod, daemon=True)
    thread_read_frame.start()
    thread_img_process.start()
    while isend!=1:
        # print("hi3\n")
        frame1 = nodeAfterBuff.get()
        cv2.imshow('ImageWindow', frame1)
        cv2.waitKey(1)


    #cv2.imshow('ImageWindow', frame)
    #cv2.waitKey(1)