# CVS - Computer Vision Studio

**Machine Vision Engineer** | 대한민국, 서울 | sejunkwon@outlook.com |

***

## 1. 레포지토리 설명
**Introduction**
* Qt Framework와 OpenCV, ONNX Runtime 을 이용해 실시간으로 영상을 GUI로 출력하면서 동시에 기계학습 모델 추론을 수행할 수 있는 GUI 프로그램
* Video Capture 과 Model Inference 는 Multi-threaded 로 분리 구현되어 프로그램 실행 중에 동적으로 추론가능
* 카메라 캘리브레이션을 수행
* 프레임이 생성되는 시간, 기계학습 모델 추론시간을 측정하는 테스트베드 프로그램

**Object**
* Computing Power가 낮은 Edge Device에서 추론할 때 소요 시간 때문에 Video Capture Frame이 Drop 되지 않도록 추론 최적화를 진행 (30fps 기준)

**Prerequisites**
* QT Framework, CUDA Toolkit, OpenCV, ONNX Runtime, GStreamer가 필요
* OpenCV는 인텔 oneAPI의 TBB와 IPP를 지원하도록 빌드되어야 함

**테스트 환경 - Laptop**
* i7 9750H 6Core 12Threads
* 32 GB Main Memory
* Nvidia Quadro T1000 Mobile Vram 4GB FP32 2.6TFLOPS

***

## 2. 구현 과정

**요구사항 파악**
* PC에 연결된 웹캠에서 프레임을 얻는다
* 추론 엔진으로 얻은 frame에서 detection 한다
* GUI를 업데이트 한다
* 멀티스레딩 구조가 필요하다 (Blocking이 없어야 한다)
* 추론엔진을 ONNX를 TensorRT, OpenVINO 등으로 교체할 수 있어야 한다
* GUI 가 Block 되거나 캡처된 프레임이 추론 시간을 기다리느라 Drop되면 안된다

**구현**
* Qt Designer 의 UIC를 사용하지 않고 직접 QWidget기반 객체를 배치하여 구현
* Qt MainWindow, FrameController, CaptureController, InferenceController 의 객체로 분리
  - 이 때 멀티스레딩은 Qt의 Object-Worker 모델을 이용, 각 Controller 객체는 QThread객체를 가지고 있으며 \
	이 스레드 객체가 생성될 때 실제 스레드를 생성하고 작업을 실행, 작업이 종료될 때 QThread객체도 같이 소멸하도록 구현 \
	Worker Object의 Thread affinity가 멀티스레딩 작업이 시작될 때 그 스레드로 변경됨
* cv::Mat Frame은 = 연산자를 사용하면 shallow copy 가 되는 점을 이용 FrameController 에서 한번만 생성하고 다른 스레드에서는 Frame의 포인터만 받아 참조하도록 구현
* Qt의 signal, slot, connect를 이용하여 프레임 캡처가 완료 된 시점과 Inference의 시작 시점을 맞춰줌
* 이후 Frame의 모든 처리가 끝나면 signal, slot으로 GUI에 신호를 보내 프레임 갱신


**분석**
* QElapsedTimer를 이용해 코드 라인의 실행시간을 측정
* ONNX Runtime은 실행하려는 기계학습 모델의 모든 커널을 한 덩어리의 계산 그래프로 만들어 실행하므로 프로파일링을 위해 Nvidia Nsight Systems를 이용
  - Nsight Systems를 사용하면 프로그램의 전체 생명주기에서 CUDA Kenrel들과 프로그램 요소들의 실행시간을 측정할 수 있음

***

## 3. 깨달은 점

* 스레드 분리 구현할 때 QMutex를 통해 GUI스레드와 CaptureWorker 스레드가 frame 멤버 변수에 접근하는 것을 통제하지 않으면 메모리가 크래시됨
  - QMutex를 통해 Concurrent하게 frame에 접근하게 함
* CaptureWorekr를 끄고 켤 수 있게 만들 때 단순히 While문으로 캡처 루프를 구현하게 되면 한 스레드가 영원히 While문 멤버 함수를 돌게 되므로 중간에 제어할 수가 없게 됨(Event Queue의 동작 특성 때문)
  - 그러므로 InvokeMethod를 통해 한 프레임씩 캡처를 실행하고 종료하는 과정을 연속으로 하게 하여 중간에 GUI 메인스레드에서 캡처를 종료하는 멤버함수를 실행할 수 있도록 하여 중간에 캡처 중지 및 실행 기능을 구현
* CaptureWorker를 실행하면서 InferenceWorker를 켜고 끌 수 있는 기능을 구현할 때 기본적으로 Queued Connection 은 이벤트 루프의 큐에 다음에 실행할 멤버함수를 넣어놓고 비동기적으로 Capture Engine 스레드가 다음라인으로 넘어가게 됨 \
  이 때문에 Inference가 실행될 때 캡처한 프레임이 이미 GUI에 표시 되고 나서 과거의 프레임에 대해 추론하고 영원히 추론결과가 GUI스레드에 표시되지 않는 문제가 있었음. 
  - Blocking Queued Connection 타입을 connect에 지정하여 Inference가 다 끝날 때까지 Capture Engine 스레드를 Block 하게 만들어 온전한 순서대로 프레임 캡처 -> 추론 -> GUI 에 표시 하는 Concurrent한 순서가 보장되도록 수정하니 잘 동작.
***

## 4. Subproject - 30Fps

* 웹캠 실시간 캡처에서 30Fps로 프레임이 생성될 때 이러한 프레임들이 기계학습 추론할 때 drop 되거나 block 되지 않게 하도록 추론 pipline을 수정
* 33ms라면 대략 이 시간을 주기로 cap_ >> frame 에서 프레임이 생성됨
* 그런데 여기서 추론시간이 33ms 가 넘어간다면 프레임이 지연되거나 dorp됨

**시간측정**

![screenshot](https://github.com/sejunkwonme/CVS/assets/firstresult.png)