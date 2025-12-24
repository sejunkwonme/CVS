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

## 4. Example
![screenshot](/assets/example.png)

## 5. Subproject

* 웹캠 실시간 캡처에서 30Fps로 프레임이 생성될 때 이러한 프레임들이 기계학습 추론할 때 drop 되거나 block 되지 않게 하도록 추론 pipline을 수정
* 33ms라면 대략 이 시간을 주기로 cap_ >> frame 에서 프레임이 생성됨
* 그런데 여기서 추론시간이 33ms 가 넘어간다면 프레임이 지연되거나 dorp됨

**시간측정** [us는 microsecond를 표현, ms 는 milisecond]

![screenshot](/assets/secondresult.png)

* 최초 프레임 생성 이후 YUY2 -> RGB 변환, 그리고 Crop 하는데에 1460us(1.46ms) 소요
* 이후 Inference 스레드에서 처리될 때 모델의 입력을 만드는 cv::blobFromImage에서 약 2.5~4 ms 소요
* blob생성후 Ort Session Run 시에 약 37.5~39ms 소요
* 추론후 데이터를 받아 후처리 후 cv::dnn::NMSBoxes 를 수행하는 동안 약 1000~1300us(1ms) 소요

이 시간들을 다 합하면 프레임 생성 후 43~45ms 정도 소요된다. 30Fps는 33ms 주기마다 프레임이 생성되므로 이 주기에 들어가지 않아
프레임이 밀리거나 drop된다.

**전략**

* ONNX Runtime CUDA Excution Provider는 Session Run 할 때 노드를 자동으로 감지하여 cudnn에서 인식할 수 있는 Operation은 자동으로 최적화된 커널을 만들어서 실행해 준다.
* 이렇게 만들어준 커널은 최적화의 성능마진이 매우 작은 수준으로 잘 최적화 되므로 Session 내부를 최적화 하지는 않는다.
* 생각보다 raw 이미지를 변환하고, tensor형식으로 만들어 주는 opencv api 함수들이 시간소요가 크다. 이 부분을 CUDA 커널로 직접 구현하여 지연 시간을 줄이기로 계획
* cv::dnn:NMSBoxes 가 시간소요가 많이 걸릴것으로 예상되었으나 실제로는 매우 빠르게 실행됨
* 그래도 D2H, H2D 시간을 줄이기 위해 대부분의 메모리가 Device메모리 상에서 움직이도록 구현하도록 계획

**실제 적용**

* 1단계
* 현재 상태는 웹캠에서 최초 프레임 생성 후 [Cap Overhead] -> [blob] -> [ORT Run] ->	 [Postprocess] 순서대로 진행되고 있음
* [Cap Overhead] 는 raw YUV2로 나온 이미지를 처리하는 cv::cvtColor와 cv::Crop을 포함함
* [blob]은 interleaved BGR로 정렬되어 있는 이미지를 planar RGB 로 바꿔주고, float32 [0.0:1.0] 범위로 스케일링도 해줌
  - CUDA커널을 작성하여 위의 3개의 과정을 2개의 CUDA커널로 작성후 시간 단축함
  - (Cap Overhaed 의 1.5ms + blob 의 2.5ms) 약 4에서5ms -> 총합 약 0.5에서6~ms로 단축
![screenshot2](/assets/figure2.png)

* 2단계
* 전처리 단계를 단축 하고도 현재는 여전히 ONNX Runtime에서 CPU Tensor를 받아서 추론을 진행중임 이는 Ort Session Run 의 병목현상을 불러옴
* cudamemcpy로 전처리 결과를 Host로 꺼내올 필요 없이 zero-copy로 이미지 전처리 완료되면 바로 Device memory를 참조하여 추론을 실행하도록 구현
* ort::session->run 시간 미세하게 단축됨 (약 1~2ms)
![screenshot3](/assets/figure3.png)

* 3단계
* 2단계에서 만든 전처리 cuda커널이 너무나 많은 thread들을 사용 (640 x 480 개의 약 30만개) 그래서 커널 런치 오버헤드와 실행 시간에서 병목이 발생
* YUV2 변환과 CROP, planar변환과 [0:1] 스케일링을 하나의 커널 안에서 처리하도록 수정, 그리고 내부적으로 반복문을 사용하여 훨씬 적은 thread들을 사용하게 최적화 (약1만개)
* 전처리 시간 0.5ms -> 0.05 ~ 0.1 ms 로 단축
* 어떤 수를 써도 ort::session->run 의 시간을 33ms 미만으로 줄일 수 없음
* 그래서 yolov1모델 자체를 쪼개서 onnx 파일로 만들어 뒷 단계 ort session 이 실행중인 동안 카메라 프레임이 생성되어 들어오면 전의 세션이 바로 받아 처리하도록 겹치기로 함
* 두개의 ort session 은 각각의 cuda 스트림에서 실행되어 동시에 병렬로 gpu에서 처리될 수 있음 이렇게 겹치는 만큼 성능향상을 얻을 것으로 예상
* 두번째 세션이 끝나면 결과를 가지고 렌더링 하도록 구현 두번째 세션이 실행중인동안 첫번째 세션이 바로 시작될 수 있음
* ort IoBinding을 통해 캡처 이후부터 추론 완료까지는 다시 host메모리로 나오지 않고 쭉 진행된다.
![screenshot4](/assets/thirdresult.png)
![screenshot5](/assets/preprocess.png)

* 최종적으로 평균적으로 27프레임, 상한으로는 가끔 29, 30 프레임 간격 속도로 추론결과와 렌더링이 나오게 됨. (38ms -> 33~35ms)
* 하지만 중간의 framecontroller에 2개의 cv::Mat버퍼를 가지고 동기화 없이 번갈아 가면서 capture결과를 덮어 쓰도록 했기 때문에 data race condition 발생, 화면 출력에서 가끔씩 jittering 이 보임. 이는 fps를 끌어올리기 위해 tradeoff 한 결과임.
* 그래도 더 이상 줄일 수 없었던 추론시간 자체를 줄여볼 수 있었음.
* 이전까지는 캡처 중에 추론을 껏다 켤 수 있었지만 원 버튼으로 만들어 실행순서를 보장해야 했음

* qt와 멀티스레딩을 이용한 각 객체간의 데이터 전달과 생명주기, 그리고 CUDA커널을 통한 이미지 처리 대폭 시간감축, 더 나아가서는 통합 파이프라인의 설계에 관해 생각해볼 수 있는 시간이 되었음.