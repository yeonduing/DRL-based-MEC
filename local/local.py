import cv2
import io
import socket
import struct
import time
import pickle
import zlib

import numpy as np
import math

import queue
import threading
from threading import Thread

############## 전역 변수 ##############
# 노드, 에지가 한 프레임을 처리하는데 걸리는 추정 시간
node_time = 0.001
edge_time = 0.002
# 지연 시간을 계산할 때 사용할 상수
a = 0.6

############## 이미지 처리 변수 ##############
# 이미지 사이즈(cols, rows)를 초기화
width = 1920;
height = 1080;

#width = 180
#height = 176

w = width
h = height
K = np.array([[w, 0, w / 2], [0, h, h / 2], [0, 0, 1]])

# 처리할 프레임 개수
img_cnt = 0

#q_max size
q_max = 5000
# 해당 순서의 프레임을 노드와 에지중 어느곳에서 처리 했는지 표시
procSeq = queue.Queue(q_max)
# 노드, 에지에서 영상처리를 해야하는 Mat 저장
nodeBeforeBuff = queue.Queue(q_max)
edgeBeforeBuff = queue.Queue(q_max)
# 노드, 에지에서 영상처리를 마친 Mat 저장
nodeAfterBuff = queue.Queue(q_max)
edgeAfterBuff = queue.Queue(q_max)

####### 통신 변수 #######
# ip addr
ip_konkuk = '192.168.86.59' # 공대
ip_konkuk_univ = '192.168.37.255' # 공대
ip_home = '192.168.0.3'
ip_phone = '172.20.10.2'
ip_server = '114.70.22.26'
ip = 'ip_server'
clnt_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
clnt_sock.connect((ip, 8485))
connection = clnt_sock.makefile('wb')
# 통신 끝을 알리는 메세지
msg_stop = 'stop'

####### opencv video 변수 #######
cam = cv2.VideoCapture('IMG_4302.MOV')
cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]

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
    global node_time
    while True:
        start_time = time.time()
        frame = nodeBeforeBuff.get()
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
                          
        cur_node_time = time.time() - start_time
        
        node_time = node_time * a + cur_node_time * (1 - a)
        nodeAfterBuff.put(frame)
        print('img_mod')
        # cv2.imshow('ImageWindow', frame)
        # cv2.waitKey(1)
### img_mod END

############## Socket Communicate 함수 ##############
### recv_img : edge로부터 처리한 영상을 받는다.
def recv_img():
    data = b""
    # clnt_sock.send(msg_go.encode('utf-8'))
    #calcsize @시스템에 따름. = 시스템에 따름 < 리틀 엔디안 > 빅 엔디안 !네트워크(빅 엔디안)
    #원본 >L
    payload_size = struct.calcsize(">L")
    # print("payload_size: {}".format(payload_size))
    
    while len(data) < payload_size:
        # print("Recv: {}".format(len(data)))
        data += clnt_sock.recv(4096)
    
    # print("Done Recv: {}".format(len(data)))
    packed_msg_size = data[:payload_size]
    data = data[payload_size:]
    msg_size = struct.unpack(">L", packed_msg_size)[0]
    # print("msg_size: {}".format(msg_size))
    while len(data) < msg_size:
        data += clnt_sock.recv(4096)
    frame_data = data[:msg_size]
    data = data[msg_size:]

    frame = pickle.loads(frame_data, fix_imports=True, encoding="bytes")
    frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
    # cv2.imshow('ImageWindow', frame)
    # cv2.waitKey(1)
    
    edgeAfterBuff.put(frame)
# frames.append(frame)

# cv2.imshow('ImageWindow', frame)
# cv2.waitKey(40)
### recv_img END

### send_img :
def send_img():
    frame = edgeBeforeBuff.get()
    result, frame = cv2.imencode('.jpg', frame, encode_param)
    data = pickle.dumps(frame, 0)
    size = len(data)
    
    # print("{}: {}".format(img_cnt, size))
    clnt_sock.sendall(struct.pack(">L", size) + data)
### send_img END

### sock_commu() :
def sock_commu():
    #time.sleep(10)
    global edge_time
    while True:
        start_time = time.time()
        send_img()
        recv_img()
        cur_edge_time = time.time() - start_time
        edge_time = edge_time * a + cur_edge_time * (1 - a)
        print('socket')
### sock_commu() END

############## Judge Algorithm 함수 ##############
def judge_algo():
    global img_cnt
    global node_time
    global edge_time
    
    while (cam.isOpened()):
        ret, frame = cam.read()
        if frame is None:
            break
        # result, frame = cv2.imencode('.jpg', frame, encode_param)

        node_latency = node_time * (nodeBeforeBuff.qsize() + 1)
        edge_latency = edge_time * (edgeBeforeBuff.qsize() + 1)
        
        if node_latency <= edge_latency:
            procSeq.put(1)
            nodeBeforeBuff.put(frame)
            print('node_time: ', node_time, 'node_latency: ', node_latency)
        else:
            procSeq.put(2)
            edgeBeforeBuff.put(frame)
            print('edge_time: ', edge_time, 'edge_latency: ', edge_latency)
            
        img_cnt += 1
    procSeq.put(0)
    cam.release()

############## Img Merger 함수 ##############
def img_merger():
    while True:
        serv_num = procSeq.get()
        if serv_num == 1:
            frame = nodeAfterBuff.get()
            #cv2.imshow('ImageWindow', frame)
            #cv2.waitKey(40)
        elif serv_num == 2:
            frame = edgeAfterBuff.get()
            #cv2.imshow('ImageWindow', frame)
            #cv2.waitKey(40)
        else:
            break

### threads_func :
if __name__ == '__main__':
    main_start_time = time.time()
    thread_img_process = threading.Thread(target = img_mod, daemon = True)
    thread_sock_commu = threading.Thread(target = sock_commu, daemon = True)
    thread_judge_algo = threading.Thread(target = judge_algo, daemon = True)
    #thread_img_merger = threading.Thread(target = img_merger)
    thread_judge_algo.start()
    thread_sock_commu.start()
    thread_img_process.start()
    #thread_img_merger.start()
    #thread_img_merger.join()
    while True:
        serv_num = procSeq.get()
        if serv_num == 1:
            frame = nodeAfterBuff.get()
            print('node에서 프린트')
        #cv2.imshow('ImageWindow', frame)
        #cv2.waitKey(40)
        elif serv_num == 2:
            frame = edgeAfterBuff.get()
            print('edge에서 프린트')
        else:
            print('끝이다')
            break
        cv2.imshow('ImageWindow', frame)
        cv2.waitKey(1)
    clnt_sock.send(msg_stop.encode('utf-8'))
    clnt_sock.close()
    main_end_time = time.time() - main_start_time
    print(main_end_time)
### threads_func END

