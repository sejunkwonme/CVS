#include "Capture.h"
#include <QElapsedTimer>
#include <QtConcurrent>
#include <QDebug>
#include "Utilities.h"

using namespace std;

Capture::Capture(int camera, QMutex* lock, QObject* parent) : running(false), cameraID(camera), videoPath(""), data_lock(lock), taking_photo(false) {
    // 1) CUDA디바이스 스캔
    int device_count = 0;
    cudaGetDeviceCount(&device_count);

    int selected_id = 0;
    for (int i = 0; i < device_count; ++i) {
        cudaDeviceProp prop{};
        if (cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
            qDebug() << "CUDA Device" << i << ":" << prop.name;
            // 필요시 더 정교한 규칙으로 선택
            if (std::string(prop.name).find("NVIDIA") != std::string::npos) {
                selected_id = i;
                break;
            }
        }
    }
    qDebug() << "Selected CUDA device:" << selected_id;

    // 2) 환경변수로 CUDA 디바이스 노출 고정
    //    (이후 ORT 내부에선 이 장치가 0번으로 보입니다)
    {
        // "1" 같은 숫자 대신 selected_id를 문자열로
        char buf[8]; _snprintf_s(buf, _TRUNCATE, "%d", selected_id);
        _putenv_s("CUDA_VISIBLE_DEVICES", buf);
    }

    // 3) ORT 세션 옵션
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    session_options.SetIntraOpNumThreads(1);
    session_options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);

    // 4) CUDA EP 등록 (이제 visible index 0이 우리가 고른 NVIDIA)
    OrtCUDAProviderOptions cuda_options{};          // 최소 초기화가 가장 호환성 높음
    cuda_options.device_id = 0;                     // 중요: 0으로 고정 (위 환경변수 때문에 NVIDIA가 0번)
    session_options.AppendExecutionProvider_CUDA(cuda_options);
    //으아우아어아어아
    // 5) 세션 생성 + 실패 시 CPU fallback
    try {
        const wchar_t* model_w = L"D:/Repo/Yolov1/yolomodel.onnx";
        ort_session = new Ort::Session(ort_env, model_w, session_options);
        qDebug() << "Session created successfully (CUDA).";
    } catch (const Ort::Exception& e) {
        qCritical() << "CUDA session creation failed:" << e.what();
        qCritical() << "→ Falling back to CPU provider.";
        // CPU 전용으로 재시도
        session_options = Ort::SessionOptions{};
        const wchar_t* model_w = L"D:/Repo/Yolov1/yolomodel.onnx";
        ort_session = new Ort::Session(ort_env, model_w, session_options);
    }

    // 6) Provider 리스트 확인 로그(디버깅용)
    for (auto& p : Ort::GetAvailableProviders())
        qDebug() << "Provider:" << QString::fromStdString(p);

    // 7) 입력/출력 이름: 포인터 수명 문제 해결
    {
        Ort::AllocatorWithDefaultOptions allocator;
        Ort::AllocatedStringPtr in_ptr = ort_session->GetInputNameAllocated(0, allocator);
        Ort::AllocatedStringPtr out_ptr = ort_session->GetOutputNameAllocated(0, allocator);
        // AllocatedStringPtr의 메모리는 포인터 소멸 시 해제됩니다.
        // 따라서 std::string으로 복사해 두고 c_str()를 보관하세요.
        input_name_str = std::string(in_ptr.get());
        output_name_str = std::string(out_ptr.get());
        input_names = { input_name_str.c_str() };
        output_names = { output_name_str.c_str() };
        qDebug() << "IO names:" << input_names[0] << "->" << output_names[0];
    }

    qDebug() << "YOLOv1 ONNX model ready.";
}

Capture::Capture(QString videoPath, QMutex* lock) : running(false), cameraID(-1), videoPath(videoPath), data_lock(lock), taking_photo(false) {
}

Capture::~Capture() {
    delete ort_session;
    ort_session = nullptr;
}

void Capture::captureLoop() {
    running = true;
    cv::VideoCapture cap(cameraID, cv::CAP_MSMF);
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    cap.set(cv::CAP_PROP_FPS, 30);

    if (!cap.isOpened()) {
        qWarning() << "Camera open failed";
        return;
    }

    cv::Mat tmp_frame;

    while (running) {
        cap >> tmp_frame;
        if (tmp_frame.empty()) break;

        // --- 중앙 크롭 (448x448) ---
        int W = tmp_frame.cols;
        int H = tmp_frame.rows;
        int x_start = (W - 448) / 2;
        int y_start = (H - 448) / 2;
        cv::Rect roi(x_start, y_start, 448, 448);
        cv::Mat cropped = tmp_frame(roi).clone();

        // --- YOLOv1 추론 ---
        detectObjects(cropped);

        // --- 신호 전송 ---
        cv::Mat rgb;
        cv::cvtColor(cropped, rgb, cv::COLOR_BGR2RGB);
        data_lock->lock();
        frame = rgb;
        data_lock->unlock();
        emit frameCaptured(&frame);
    }

    cap.release();
    running = false;
}

void Capture::takePhoto(cv::Mat& frame) {
    QString photo_name = Utilities::newPhotoName();
    QString photo_path = Utilities::getPhotoPath(photo_name, "jpg");
    cv::imwrite(photo_path.toStdString(), frame);
    emit photoTaken(photo_name);
    taking_photo = false;
}

void Capture::detectFaces(cv::Mat& frame) {
    vector<cv::Rect> faces;
    cv::Mat gray_frame;
    cv::cvtColor(frame, gray_frame, cv::COLOR_BGR2GRAY);
    classifier->detectMultiScale(gray_frame, faces, 1.3, 5);

    cv::Scalar color = cv::Scalar(0, 0, 255);
    for (size_t i = 0; i < faces.size(); i++) {
        cv::rectangle(frame, faces[i], color, 1);
    }
}

void Capture::detectObjects(cv::Mat& frame) {
    // --- 전처리 ---
    //cv::Mat rgb;
    //cv::cvtColor(frame, rgb, cv::COLOR_BGR2RGB);

    cv::Mat blob = cv::dnn::blobFromImage(
        frame,
        1.0 / 255.0,
        cv::Size(frame.cols, frame.rows),
        cv::Scalar(),
        true,
        false,
        CV_32F
    );

    const int S = 7, B = 2, C = 20;
    const int H = 448, W = 448;
    std::array<int64_t, 4> input_shape{ 1, 3, 448, 448 };

    std::vector<float> input_tensor_values(1 * 3 * H * W);

    float* blob_data = blob.ptr<float>();
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
        OrtAllocatorType::OrtArenaAllocator, OrtMemTypeDefault
    );


    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        blob_data,
        1 * 3 * 448 * 448,
        input_shape.data(),
        input_shape.size()
    );

    auto output_tensors = ort_session->Run(
        Ort::RunOptions{ nullptr },
        input_names.data(), &input_tensor, 1,
        output_names.data(), 1
    );

    float* preds = output_tensors.front().GetTensorMutableData<float>();

    // --- 디코딩 ---
    const float conf_thresh = 0.15f;  // 임계값 낮게 설정
    const int stride = (5 * B + C);   // 30

    for (int i = 0; i < S; ++i) {
        for (int j = 0; j < S; ++j) {
            const int offset = S * S;

            int   best_id = 0;
            float best_v = -std::numeric_limits<float>::infinity();

            // 클래스 확률 (20개)
            for (int clsidx = 0; clsidx < 20; ++clsidx) {
                float v = preds[(i * S + j) + (clsidx * offset)];
                if (v > best_v) {
                    best_v = v;
                    best_id = clsidx;
                }
            }

            int   class_id = best_id;
            float class_conf = best_v;


            // 두 개의 box 각각 표시
            for (int b = 0; b < B; ++b) {
                float x_cell = preds[(i * S + j) + (21 * offset)];
                float y_cell = preds[(i * S + j) + (22 * offset)];
                float w = preds[(i * S + j) + (23 * offset)];
                float h = preds[(i * S + j) + (24 * offset)];
                float conf = preds[(i * S + j) + (20 * offset)];

                float score = conf * class_conf;
                if (score < conf_thresh)
                    continue;

                // 셀 기준 → 이미지 좌표계 변환
                float x = (j + x_cell) / S;
                float y = (i + y_cell) / S;
                float box_w = w;
                float box_h = h;

                int x1 = static_cast<int>((x - box_w / 2) * frame.cols);
                int y1 = static_cast<int>((y - box_h / 2) * frame.rows);
                int x2 = static_cast<int>((x + box_w / 2) * frame.cols);
                int y2 = static_cast<int>((y + box_h / 2) * frame.rows);

                cv::Rect box(cv::Point(x1, y1), cv::Point(x2, y2));
                cv::Scalar color = cv::Scalar(0, 255 * (b == 0), 255 * (b == 1));
                cv::rectangle(frame, box, color, 1);

                // 클래스 ID와 점수 표시
                std::string label = std::to_string(class_id) + ":" + cv::format("%.2f", score);
                cv::putText(frame, label, cv::Point(x1, y1 - 3),
                    cv::FONT_HERSHEY_SIMPLEX, 0.3, color, 1);
            }
        }
    }
}
