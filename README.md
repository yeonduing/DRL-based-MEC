# 영상처리를 위한 강화학습 기반 에지 컴퓨팅
> Edge Computing based on Deep Reinforcement Learning for Image Processing

본 프로젝트의 주제는 강화 학습을 적용한 에지 컴퓨팅(Edge Computing) 기술이 실시간 처리를 필요로 하는 데이터를 얼마나 잘 처리 하는지 보여주는 것이다. 이에 대한 시나리오로 많은 컴퓨팅 자원을 요구하는 영상 편집을 골랐다. 로컬(Local)에 낮은 컴퓨팅 자원을 가진 컴퓨터를 두고, 에지(Edge)에 높은 컴퓨팅 자원을 가진 컴퓨터를 두어 낮은 레이턴시(Latency)로 실시간 영상 처리하는 것을 보여주는 것이 프로젝트의 목표이다.

## System Model

시스템은 Local과 Edge로 나뉜다. Local의 경우 Computation Resource Selection, Socket Communication, Image Processing과 Image Merger 총 네 부분으로 나뉜다. Edge의 경우 Socket Communication과 Image Processing 총 두 부분으로 나뉘며, Local과 Edge 모두 각 부분은 스레드로 분리하여 동시에 작동할 수 있도록 하였다. 

![system model](/images/system-model.png )
<div style="text-align: center">fig. 1 The DRL-based system architecture</div>

### Computation Resource Allocation

한 프레임이 처리 대기버퍼에서 나와 재생 대기 버퍼에 들어가기까지의 시간을 직전 처리 시간이라 하고 이를 Local 과 Edge 각각에 대하여 구한다. 두 수를 활용하여 주어진 프레임을 어디에서 처리하는 것이 빠른지 판단한다. 본 프로젝트에서는 판단 알고리즘에 대해 두 가지 방법으로 실험하였다.

**이전 값을 통한 예측 알고리즘**
Local과 Edge 각각에 대하여 기존 평균 처리 시간과 직전 처리 시간을 적절한 비율로 더하여 평균 처리 시간을 계산한다. 각각의 평균 처리 시간과 대기 중인 프레임 수를 비교하여 현재 주어진 프레임을 어디에서 처리하는 것이 빠른지 판단한다.
$$
{이전 평균 처리 시간 \times a + 직전 처리 시간 \times ( 1 – a )}\ (0 < a < 1)\\ (1)
$$

**DRL을 적용한 알고리즘**
현재 state $s_t$의 action $a_t$에 대한 Q 값 $Q(s_t)$는 다음과 같다. $$Q(s_t) = r_t\ +\ γ maxQ(s_{t+1})\\ (2)$$ $r_t$는 현재 받는 reward값이고, $γ$는 discount factor 이다. State는 [Local queue , Edge queue] 로 정의하였다. Reward는 frame이 queue에 들어가서 처리가 완료되기 까지의 시간으로 정한다. Action 은 어느 곳으로 가는지에 대한 값, γ값은 일반적인 예시에서 쓰는 0.9를 사용한다.[1]

### Socket Communication

Local과 Edge는 Socket을 통해 통신한다. 프레임에 해당하는 Mat변수(영상)를 uchar형태로 변환하여 배열에 저장하고 Server에 해당하는 PC의 IP 주소와 Port번호를 지정해 해당 배열을 전송한다. 데이터 송수신 과정에서 송신과 수신의 속도 차에 의한 버퍼 관리를 위하여 TCP 통신 프로토콜을 따라 소켓 프로그래밍을 했다.

### Image Processing

아래 자료는 극좌표계에서의 좌표로 계산하여 휘어진 이미지를 보여준다. 사용자가 처리 후에 받게 될 결과이며, 극좌표계에서의 좌표는 모니터 화면과 맞지 않으므로 fig.2 와 같이 평면 직사각형에 투영하여 전송한다. 이는 [2]의 코드를 참고하여 작성하였다.

![result img](/images/processed-img.png )
<div style="text-align: center">fig. 2 The result image</div>

### Imgae Merger

하나의 영상을 프레임 단위로 나누어 Local과 Edge 두 곳에서 처리하므로 사용자에게 송출할 때에는 순서를 다시 맞추는 작업이 필요하다. 이를 위해 Computation Resource Allocation 부분에서는 어느 곳에서 처리할지 분배해주면서 그것을 기록한다. 이는 전역 변수인 preSeq에 저장되며 queue 자료형이므로 입력한 순서대로 출력되어 프레임의 순서를 맞출 수 있다. 

## Performance Evaluation

### Experimental Settings
>Local: i5 3세대 U버전 CPU 
 Edge: i5 - 7500

영상 처리를 CPU로만 하기에 컴퓨팅 파워를 CPU 성능으로 생각하였고, Cinebench R15 툴 기준 멀티코어 성능2배 차이, 싱글코어 성능은 1.5배 정도의 성능 차이를 가지고 있다. 랜선을 이용한 유선 상의 비교는 평균치를 이용한 방법과 DRL을 이용한 방법에서 굉장히 동적인latency의 변화에 대해 대응하는 것을 보기에는 적절치 않다고 생각하여 일반적인 54Mbps 속도의 Wi-Fi환경에서 실험을 진행하였다.
3가지 해상도(480P, 720P)의 뮤직비디오 영상을 3,4 번 정도 실험하여 Latency들의 평균값을 구하였다.

### Result Analysis
프로젝트의 목적은 계산(영상처리)을 에지 컴퓨팅을 통해 향상시키는 것이고, 나아가서는 강화학습을 적용하여 매 시각 다양하게 변화하는 네트워크 환경에 대해 Computing Resource Selection을 적절히 하여 임의로 설정한 parameter를 통한 결과보다 더 좋은 결과를 얻는 것이다. 아래의 두 사진은 각각 Local에서만 영상처리를 했을 때, 임의의 parameter를 설정하여 평균치를 이용한 분배와 학습을 이용한 분배의 비교를 나타낸 Empirical CDF이다. 세로는 확률, 가로는 프레임과 프레임 사이의 latency이다. 프레임 사이의 latency가 작을수록 영상의 frame이 옳은 순서로 더 빠르게 처리(재생)되어 더 높은 성능을 가진다고 볼 수 있다. 즉, Computing Resource Selection이 옳게 이루어져 Local혼자 처리할 때 보다 높은 성능향상을 보인다.

| ![analysis local only](/images/analysis-local-only.png )| ![analysis local only](/images/analysis-DRL.png ) | 
| --- | --- |
|<div style="text-align: center">fig. 3 The result analysis when processing in the local server only</div> | <div style="text-align: center">fig. 4 The result analysis when processing in the local server only</div> |

## Conclusion

Edge Computing을 이용하여 Computing 작업을 수행했을 때, Local 단독으로 충분히 빠르게 처리할 수 있는 작업은 단독으로 처리하는 것이 나으나, 그렇지 않은 경우에는 항상 Edge Computing을 적용한 경우가 좋았다(Latency가 굉장히 높아진다거나 하는 대륙-대륙 간 연결이나 Wi-Fi가 자주 끊기고 연결되는 경우를 제외한 일반적인 환경에서). 특히 DRL을 적용하였을 때가 평균치를 이용한 경우보다 좋았는데 Edge Computing과 DRL을 적용한다면 평균Latency를 거의 0.1~0.2 초 정도 줄일 수 있었다.

## Reference

[1] J. Wang, L. Zhao, J. Liu, and N. Kato, “Smart resource allocation for mobile edge computing: A deep reinforcement learning approach,” IEEE Trans. Emerg. Topics Comput., to be published. doi: 10.1109/TETC.2019.2902661.
[2] “Warp an image to cylindrical coordinates for cylindrical panorama stitching, using Python OpenCV,” [Online]. Available: https://gist.github.com/royshil/0b21e8e7c6c1f46a16db66c384742b2b