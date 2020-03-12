# -*- coding: utf-8 -*-

import numpy as np

import socket
import sys

import queue

import cv2
import pickle

import struct ## new
import zlib
import math
import multiprocessing
import threading
from threading import Thread

############## 전역 변수 ##############
### __init__ :


############## 이미지 처리 변수 ##############
# 에지에서 영상처리를 해야하는 Mat 저장
#q_max size
q_max = 5000
edgeBeforeBuff = queue.Queue(q_max)
# 에지에서 영상처리를 마친 mat 저장
edgeAfterBuff = queue.Queue(q_max)

# 이미지 사이즈(cols, rows)를 초기화
width = 1920;
height = 1080;

w = width
h = height
K = np.array([[w, 0, w / 2], [0, h, h / 2], [0, 0, 1]])

encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]

# 처리할 프레임 개수
img_cnt = 0

####### 통신 변수 #######
HOST = ''
PORT = 8485
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
print('Socket created')
recv_addr = (HOST, PORT)
sock.bind(recv_addr)
print('Socket bind complete')
sock.listen(0)
print('Socket now listening')
clnt_sock, clnt_addr = sock.accept()
### __init__ END

############## ImageProcess 함수 ##############
### convert_pt :
def convert_pt(x, y, w, h):
    px = x-w/2
    py = y-h/2
    #곡률
    f= w;
    #값을 바꿀수록 덜 휘거나 더 휨
    r= w;
    
    omega = w/2
    z0 = f- math.sqrt(r*r - omega * omega)
    
    zc = (2*z0 + math.sqrt(4*z0*z0 - 4*(px*px/(f*f)+1)*(z0*z0-r*r))) / (2 * (px*px/(f*f) + 1))
    final_x = px*zc/f + w/2
    final_y = py*zc/f + h/2
    
    return final_x, final_y
### convert_pt END

### img_mod : frame을 VR영상으로 변환
def img_mod():
    # royshil's cylindricalWarping.py
    # https://gist.github.com/royshil/0b21e8e7c6c1f46a16db66c384742b2b
    while True:
        frame = edgeBeforeBuff.get()
        """This function returns the cylindrical warp for a given image and intrinsics matrix K"""
        h_, w_ = frame.shape[:2]
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
        
        frame_rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)  # for transparent borders...
        # warp the image according to cylindrical coords
        frame = cv2.remap(frame_rgba, B[:, :, 0].astype(np.float32), B[:, :, 1].astype(np.float32), cv2.INTER_AREA,
                         borderMode=cv2.BORDER_TRANSPARENT)
        edgeAfterBuff.put(frame)
        #cv2.imshow('ImageWindow', frame)
        #cv2.waitKey(1)
### img_mod END


############## Socket Communicate 함수 ##############
### recv_img : client로부터 처리할 영상을 받는다.
def recv_img():
    data = b""
    # clnt_sock.send(msg_go.encode('utf-8'))
    #calcsize @시스템에 따름. = 시스템에 따름 < 리틀 엔디안 > 빅 엔디안 !네트워크(빅 엔디안)
    #원본 >L
    payload_size = struct.calcsize(">L")
    print("payload_size: {}".format(payload_size))

    while len(data) < payload_size:
        print("Recv: {}".format(len(data)))
        data += clnt_sock.recv(4096)
    
    print("Done Recv: {}".format(len(data)))
    packed_msg_size = data[:payload_size]
    data = data[payload_size:]
    msg_size = struct.unpack(">L", packed_msg_size)[0]
    print("msg_size: {}".format(msg_size))
    # 소켓통신 끝내기
    if msg_size == 1937010544:
        clnt_sock.close()
        return True

    while len(data) < msg_size:
        data += clnt_sock.recv(4096)
    frame_data = data[:msg_size]
    data = data[msg_size:]

    frame = pickle.loads(frame_data, fix_imports=True, encoding="bytes")
    frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
        
    edgeBeforeBuff.put(frame)
    return False
# frames.append(frame)

# cv2.imshow('ImageWindow', frame)
# cv2.waitKey(40)
### recv_img END

### send_img :
def send_img():
    frame = edgeAfterBuff.get()
    result, frame = cv2.imencode('.jpg', frame, encode_param)
    data = pickle.dumps(frame, 0)
    size = len(data)
    
    print("{}: {}".format(img_cnt, size))
    clnt_sock.sendall(struct.pack(">L", size) + data)
### send_img END

### sock_commu() :
def sock_commu():
    while True:
        ifEnd = recv_img()
        if ifEnd:
            print('sock_commu end 1')
            break
        send_img()
### sock_commu() END

### threads_func :
if __name__ == '__main__':
    thread_img_process = threading.Thread(target = img_mod, daemon = True)
    thread_sock_commu = threading.Thread(target = sock_commu)
    thread_img_process.start()
    thread_sock_commu.start()
    thread_sock_commu.join()
    print('sock_commu end 2')
    sock.close()
    print('sock close')
### threads_func END
