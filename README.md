# CVS - Computer Vision Studio

**Computer Vision Engineer** | 대한민국, 서울 | sejunkwon@outlook.com |

***

## 1. 레포지토리 설명
**Introduction**
* Qt 프레임워크와 OpenCV, ONNX Runtime 을 이용해 실시간으로 영상을 GUI로 출력하면서 동시에 기계학습 모델 추론을 수행할 수 있는 GUI 프로그램입니다.
* Video Capture 과 Model Inference 는 Multi-threaded 로 분리 구현되어 프로그램 실행 중에 동적으로 중지, 실행, 교체가 가능합니다.
* 영상에 대해 사진을 촬영하거나 동영상 녹화를 하여 저장할 수 있습니다. (구현예정)
* 카메라 캘리브레이션을 진행할 수 있습니다. (구현예정)
* 파일 익스플로러를 통해 파일을 선택하여 카메라 캘리브레이션에 사용합니다. (구현예정)
* 설정을 Sqlite 데이터베이스에 저장 가능하며 여러 카메라의 설정을 CRUD로 관리 가능합니다. (구현예정)

**Prerequisites**
* 개발을 위해서는 QT 프레임워크, OpenCV, ONNX Runtime, GStreamer가 필요합니다.
* OpenCV는 인텔 oneAPI의 TBB와 IPP를 지원하도록 빌드되어야 합니다. 

**Computational Power Specification**
* i7 265K 20Core 20Threads
* 96 GB Main Memory
* RTX 3090 Founders Edition

***

## 2. 구현 과정

**요구사항 파악**
* PC에 연결된 웹캠에서 프레임을 얻는다
* 추론 엔진으로 얻은 frame에서 detection 한다
* GUI를 업데이트 한다
* 2개 이상의 설정을 저장하고 불러올 수 있어야 한다
* 멀티스레딩 구조가 필요하다 (Blocking이 없어야 한다)
* 추론엔진을 ONNX를 TensorRT, OpenVINO 등으로 교체할 수 있어야 한다
* 카메라를 2개 이상 장착해도 문제없이 작동할 수 있어야 한다

**Process**
* Qt의 Ui 파일을 사용하지 않고 직접 QBoxLayout을 통해 QWidget들을 계속 쌓아나가는 방식으로 GUI를 구현하였습니다. 헤더파일에 위젯 객체가 명시되지 않아 가독성이 떨어졌고 컴파일 속도도 느려서 이렇게 했습니다.
* Qt MainWindow 객체와 CaptureEngine 객체를 다른 스레드에서 구동되도록 분리 구현하였습니다. 이렇게 하지 않으면 GUI 메인스레드는 영상 재생 외의 이벤트 루프 처리 작업을 할 수가 없습니다. (GUI가 Blocking 되어 멈추게 됩니다.)
* CaptureEngine 객체와 InferenceEngine 객체도 또한 분리구현하여 CaptureEngine 객체가 Blocking 되지 않도록 했습니다.
* Qt에서 제공되는 Model, View 구조를 통해 파일 익스플로러와 설정 창을 구현했습니다.

***

## 3. 깨달은 점

* 스레드 분리 구현할 때 QMutex를 통해 GUI스레드와 CaptureEngine 스레드가 frame 멤버 변수에 접근하는 것을 통제하지 않으면 크래시 되거나 화면이 이상하게 나옵니다. QMutex를 통해 Concurrent하게 frame에 접근하게 해야 올바르게 캡처 화면이 표시됩니다.
* CaptureEngine을 끄고 켤 수 있게 만들 때 단순히 While문으로 캡처 루프를 구현하게 되면 한 스레드가 영원히 While문 멤버 함수를 돌게 되므로 중간에 제어할 수가 없게 됩니다. 그러므로 InvokeMethod를 통해 한 프레임씩 캡처를 실행하고 종료하는 과정을 연속으로 하게 하여 중간에 GUI 메인스레드에서 캡처를 종료하는 멤버함수를 실행할 수 있도록 하여 중간에 캡처 중지 및 실행 기능을 구현하였습니다.
* CaptureEngine을 실행하면서 InferenceEngine을 켜고 끌 수 있는 기능을 구현할 때 기본적으로 Queued Connection 은 이벤트 루프의 큐에 다음에 실행할 멤버함수를 넣어놓고 비동기적으로 Capture Engine 스레드가 다음라인으로 넘어가게 되는데 이 때문에 Inference가 실행될 때 캡처한 프레임이 이미 GUI에 표시 되고 나서 과거의 프레임에 대해 추론하고 영원히 추론결과가 GUI스레드에 표시되지 않는 문제가 있었습니다. Blocking Queued Connection 타입을 connect에 지정하여 Inference가 다 끝날 때까지 Capture Engine 스레드를 Block 하게 만들어 온전한 순서대로 프레임 캡처 -> 추론 -> GUI 에 표시 하는 Concurrent한 순서가 보장되도록 수정하니 잘 동작했습니다.
***
