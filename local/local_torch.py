import collections
import random
import torch

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
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from threading import Thread

############## 전역 변수 ##############
# 노드, 에지가 한 프레임을 처리하는데 걸리는 추정 시간


node_time = 0.0
edge_time = 0.0
# 지연 시간을 계산할 때 사용할 상수
a = 0.6
node_time_p = 0
edge_time_p = 0

start2 = 0
start3 = 0
############## 이미지 처리 변수 ##############
# 이미지 사이즈(cols, rows)를 초기화
width = 2560
height = 1600

# width = 1920
# height = 1080

w = width
h = height
K = np.array([[w, 0, w / 2], [0, h, h / 2], [0, 0, 1]])

# 처리할 프레임 개수
img_cnt = 0

# q_max size
q_max = 5000
# 해당 순서의 프레임을 노드와 에지중 어느곳에서 처리 했는지 표시
procSeq = queue.Queue(q_max)
procSeq2 = queue.Queue(q_max)
# 노드, 에지에서 영상처리를 해야하는 Mat 저장
nodeBeforeBuff = queue.Queue(q_max)
edgeBeforeBuff = queue.Queue(q_max)
# 노드, 에지에서 영상처리를 마친 Mat 저장
nodeAfterBuff = queue.Queue(q_max)
edgeAfterBuff = queue.Queue(q_max)

nodeRewardBuff = queue.Queue(q_max)
edgeRewardBuff = queue.Queue(q_max)
nodeRewardAfterBuff = queue.Queue(q_max)
edgeRewardAfterBuff = queue.Queue(q_max)

####### 통신 변수 #######
# ip addr
ip_server = '114.70.22.26'
ip_han = '114.70.21.240'
ip_home = '192.168.0.3'
ip_phone = '172.20.10.2'
ip = ip_server
clnt_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
clnt_sock.connect((ip, 8485))
connection = clnt_sock.makefile('wb')
# 통신 끝을 알리는 메세지
msg_stop = 'stop'

####### opencv video 변수 #######
cam = cv2.VideoCapture('jk_l2.mov')
cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]


############## ImageProcess 함수 ##############
### convert_pt :
def convert_pt(x, y, w, h):
    px = x - w / 2
    py = y - h / 2
    # 곡률
    f = w;
    # 값을 바꿀수록 덜 휘거나 더 휨
    r = w;

    omega = w / 2
    z0 = f - math.sqrt(r * r - omega * omega)

    zc = (2 * z0 + math.sqrt(4 * z0 * z0 - 4 * (px * px / (f * f) + 1) * (z0 * z0 - r * r))) / (
            2 * (px * px / (f * f) + 1))
    final_x = px * zc / f + w / 2
    final_y = py * zc / f + h / 2

    return final_x, final_y


### convert_pt END

### img_mod : frame을 VR영상으로 변환
def img_mod():
    global node_time
    global node_time_p
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

        cur_node_time = time.time()
        nodeRewardAfterBuff.put(cur_node_time)
        cur_node_time -= start_time
        node_time_p = cur_node_time
        if node_time != 0.0:
            node_time = node_time * a + cur_node_time * (1 - a);
        else:
            node_time = cur_node_time
        nodeAfterBuff.put(frame)
        # cv2.imshow('ImageWindow', frame)
        # cv2.waitKey(1)


### img_mod END

############## Socket Communicate 함수 ##############
### recv_img : edge로부터 처리한 영상을 받는다.
def recv_img():
    data = b""
    # clnt_sock.send(msg_go.encode('utf-8'))
    # calcsize @시스템에 따름. = 시스템에 따름 < 리틀 엔디안 > 빅 엔디안 !네트워크(빅 엔디안)
    # 원본 >L
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
    # time.sleep(10)
    global edge_time
    global edge_time_p
    while True:
        start_time = time.time()
        send_img()
        recv_img()
        cur_edge_time = time.time()
        edgeRewardAfterBuff.put(cur_edge_time)
        # cur_edge_time = time.time() - start_time
        edge_time_p = cur_edge_time
        if edge_time != 0.0:
            edge_time = edge_time * a + cur_edge_time * (1 - a)
        else:
            edge_time = cur_edge_time


### sock_commu() END

learning_rate = 0.005
gamma = 0.9
buffer_limit = 10000
batch_size = 1


class ReplayBuffer:
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, trans):
        self.buffer.append(trans)

    def sample(self, n):  # n 만큼 랜덤하게 뽑음
        # mini_batch = random.sample(self.buffer, n) #n > 1 일 때
        mini_batch = self.buffer.popleft() #n == 1 일 때
        state1_lst, act1_lst, rew1_lst, state_prime_lst = [], [], [], []

        # for trans in mini_batch:
        state1, act1, rew1, state_prime = mini_batch
           # state1, act1, rew1, state_prime = trans
        state1_lst.append(state1)
        act1_lst.append([act1])
        rew1_lst.append([rew1])
        state_prime_lst.append(state_prime)
        # check_done_lst.append([check_done])

        return state1_lst, act1_lst, rew1_lst, state_prime_lst
        # return torch.tensor(state1_lst, dtype=torch.float), torch.tensor(act1_lst), torch.tensor(
        #     rew1_lst), torch.tensor(state_prime_lst, dtype=torch.float)  # , torch.tensor(check_done_lst)

    def size(self):
        return len(self.buffer)


class Qnet(nn.Module):

    def __init__(self):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(2, 256)  # 2 -> 256
        self.fc2 = nn.Linear(256, 256)  # 256 -> 2
        self.fc3 = nn.Linear(256, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def sample_action(self, obs, epsilon):  # epsilon greedy 를 하기 위해
        out = self.forward(obs)
        coin = random.random()
        if coin < epsilon:  # random 하게 생성한 coin 변수가 epsilon 보다 작으면 explore
            return random.randint(0, 1)
        else:  # 그렇지 않으면 exploit
            # print("exploit")
            return out.argmax().item()

def train(q, q_target, memory, optimizer):
    s, aa, r, s_prime = memory.sample(batch_size)
    s = torch.tensor(s, dtype=float)
    aa = torch.tensor(aa)
    r = torch.tensor(r)
    s_prime = torch.tensor(s_prime, dtype=float)

    q_out = q(s.float())
    # print("q_out : " , q_out)
    q_a = torch.gather(q_out, 1, aa).unsqueeze(1)
    # q_a = torch.gat
    # print("aa : " , q_a)
    # q_a = q_out.gather(1, a)  # 실제 취한 action만 사용
    #max_q_prime = q_target(s_prime.float()).max(1)[0].unsqueeze(1)
    max_q_prime = q_target(s_prime.float()).max(1)[0].unsqueeze(1)
    # print("r : " , r)
    # print("max_q_prime : " , max_q_prime)
    target = r + gamma * max_q_prime  # * done_mask
    # print("target : " , target)
    loss = F.smooth_l1_loss(q_a, target)

    # print(q_target(s_prime.float()))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def getObs(nodeBNum, edgeBNum):
    # obs = torch.tensor([nodeBNum, edgeBNum])
    if nodeBNum is None:
        nodeBNum = 0
    if edgeBNum is None:
        edgeBNum = 0
    obs = np.array([nodeBNum, edgeBNum])
    return obs


############## Judge Algorithm 함수 ##############
def judge_algo():
    global img_cnt  # 처리할 프레임 개수
    global node_time  # 1 frame을 처리하는데 걸리는 예상시간
    global edge_time  # 1 frame을 처리하는데 걸리는 예상시간

    q = Qnet()
    q_target = Qnet()
    q_target.load_state_dict(q.state_dict())  # q 에서 q_target 으로 복제
    memory = ReplayBuffer()
    optimizer = optim.Adam(q.parameters(), lr=learning_rate)  # q만 update, q_target은 q에서 복제

    NodeSeq = queue.Queue(q_max)
    EdgeSeq = queue.Queue(q_max)


    sarr, aarr, rarr, sparr = [],[],[], []

    num = 0
    r_count = 0
    r_count2 = 0
    # while (cam.isOpened()):
    while(cam.isOpened()):
        ret, frame = cam.read()
        num += 1

        if frame is None:
            break
            # result, frame = cv2.imencode('.jpg', frame, encode_param)
            # 여기부터



       # edge_latency = edge_time * (edgeBeforeBuff.qsize() + 1)

        # if node_latency < edge_latency:
        # procSeq.put(1)
        # nodeBeforeBuff.put(frame)
        # else:
        #     procSeq.put(2)
        #     edgeBeforeBuff.put(frame)


        epsilon = max(0.01, 0.08 - 0.01 * (num / 200))

        obs = getObs(nodeBeforeBuff.qsize(), edgeBeforeBuff.qsize())  # current state
        # print("nodeQsize : ", nodeBeforeBuff.qsize())
        # print("edgeQsize : ", edgeBeforeBuff.qsize())
        sarr.append(obs)
        # print("state: ", obs)
        # print("sarr pop : ",sarr.pop())

        # if num <= 1000:
        a = q.sample_action(torch.tensor(obs).float(), epsilon)  # chosen action by current state
        aarr.append(a)
        #print("action : ",a)
        if a == 0:
            procSeq.put(1)
            procSeq2.put(1)
            NodeSeq.put(num - 1)
            # print(NodeSeq.get())
            nodeRewardBuff.put(time.time())
            nodeBeforeBuff.put(frame)
            # node_latency = node_time * (nodeBeforeBuff.qsize()) + node_time_p
            # node_latency = start3
            if nodeRewardAfterBuff.qsize() != 0:
                r = -(nodeRewardAfterBuff.get() - nodeRewardBuff.get())
                # print("i am node r : ", r)
            # if r != 0:
                # print("reward start")
                rarr.insert(NodeSeq.get(), [r])
                # print("state : ", sarr[0])
                s1 = torch.tensor(sarr[0], dtype=float)
                q1 = q(s1.float())
                # print("now q : ", q1)
                r_count += 1
            s_prime = getObs(nodeBeforeBuff.qsize(), edgeBeforeBuff.qsize())
            sparr.append(s_prime)

        else:
            procSeq.put(2)
            procSeq2.put(2)
            EdgeSeq.put(num - 1)
            # print(EdgeSeq.get())
            edgeRewardBuff.put(time.time())
            edgeBeforeBuff.put(frame)
            # edge_latency = edge_time * (edgeBeforeBuff.qsize()) + edge_time_p
            if edgeRewardAfterBuff.qsize() != 0:
                r = -(edgeRewardAfterBuff.get() - edgeRewardBuff.get())
                # print("i am edge r : ", r)
            # if r != 0:
                # print("reward start")
                rarr.insert(EdgeSeq.get(), [r])
                # print("state : ", sarr[0])
                s2 = torch.tensor(sarr[0], dtype=float)
                q2 = q(s2.float())
                # print("now q : ", q2)
                r_count2 += 1
            s_prime = getObs(nodeBeforeBuff.qsize(), edgeBeforeBuff.qsize())
            sparr.append(s_prime)
        # if r is None:
        #     print("reward is none")
        # while r is None:

        # node_latency = node_time * (nodeBeforeBuff.qsize() + 1)
        # edge_latency = edge_time * (edgeBeforeBuff.qsize() + 1)
        # print("here is error")

        # if node_latency != 0:
        #     print("into node_latency")
        #     rarr.insert(NodeSeq.get(), -node_latency)
        #     r_count += 1
        #
        # if edge_latency != 0:
        #     print("into edge_latency")
        #     rarr.insert(EdgeSeq.get(), -edge_latency)
        #     r_count += 1
        # print("here is second error")
        if r_count != 0 and r_count2 != 0:
            if procSeq2.get() == 1:
                r_count -= 1
            if procSeq2.get() == 2:
                r_count2 -= 1

            # print("rarr size: ",len(rarr))
            # print("start")
            # print("sarr : " , sarr.pop(0))
            # print("aarr : " , aarr.pop(0))
            # print("rarr : " , rarr.pop(0))
            # print("sparr : " , sparr.pop(0))
            spop = sarr.pop(0)
            # print("state ok")
            apop = aarr.pop(0)
            # print("action ok")
            rpop = rarr.pop(0)
            # print("reward ok")
            sppop = sparr.pop(0)
            # print("s prime ok")
            memory.put((spop, apop, rpop, sppop))

            train(q, q_target, memory, optimizer)
            q_target.load_state_dict(q.state_dict())

        #memory.put((obs, a, r, s_prime))

        # if num > 1000:
        #train(q, q_target, memory, optimizer)

        # if (a == 0):
        #     node_latency = node_time * (nodeBeforeBuff.qsize() + 1)
        #     procSeq.put(1)
        #     nodeBeforeBuff.put(frame)
        #     done_mask = 0.0 if frame is None else 1.0
        #     memory.put((frame, a, node_latency / 100.0, f_prime, done_mask))
        #     node_score += node_latency
        #
        # else:
        #     edge_latency = edge_time * (edgeBeforeBuff.qsize() + 1)
        #     procSeq.put(2)
        #     edgeBeforeBuff.put(frame)
        #     done_mask = 0.0 if frame is None else 1.0
        #     memory.put((frame, a, edge_latency / 100.0, f_prime, done_mask))
        #     edge_score += edge_latency

        # if memory.size() > 2000:
        #     train(q, q_target, memory, optimizer)

        # if num % 50 == 0 and num != 0:

        #q_target.load_state_dict(q.state_dict())

        # print(
        #     "# of frame : {}, avg node_latency : {:.1f}, avg edge_latency : {:.1f}, buffer size : {}, epsilon : {:.1f}%".format(
        #         num, node_score / 50, edge_score / 50, memory.size(), epsilon * 100))
        #
        # node_score = 0.0
        # edge_score = 0.0

        img_cnt += 1
    # 여기까지

    procSeq.put(0)
    cam.release()

# def getReward():
#     node_latency = node_time * (nodeBeforeBuff.qsize() + 1)
#     edge_latency = edge_time * (edgeBeforeBuff.qsize() + 1)



############## Img Merger 함수 ##############
def img_merger():
    while True:
        serv_num = procSeq.get()
        if serv_num == 1:
            frame = nodeAfterBuff.get()
            # cv2.imshow('ImageWindow', frame)
            # cv2.waitKey(40)
        elif serv_num == 2:
            frame = edgeAfterBuff.get()
            # cv2.imshow('ImageWindow', frame)
            # cv2.waitKey(40)
        else:
            break


### threads_func :
if __name__ == '__main__':
    main_start_time = time.time()
    thread_img_process = threading.Thread(target=img_mod, daemon=True)
    thread_sock_commu = threading.Thread(target=sock_commu, daemon=True)
    thread_judge_algo = threading.Thread(target=judge_algo, daemon=True)
    # thread_img_merger = threading.Thread(target = img_merger)
    thread_judge_algo.start()
    thread_img_process.start()
    thread_sock_commu.start()

    # thread_img_merger.start()
    # thread_img_merger.join()
    while True:
        serv_num = procSeq.get()
        if serv_num == 1:
            frame = nodeAfterBuff.get()
        # cv2.imshow('ImageWindow', frame)
        # cv2.waitKey(40)
        elif serv_num == 2:
            frame = edgeAfterBuff.get()
        else:
            break
        cv2.imshow('ImageWindow', frame)
        cv2.waitKey(1)
    clnt_sock.send(msg_stop.encode('utf-8'))
    clnt_sock.close()
    main_end_time = time.time() - main_start_time
    print(main_end_time)
### threads_func END