#include <QDebug>

#include "Utilities.h"
#include "Inference.h"

Inference::Inference(QMutex* lock, QObject* parent)
: ort_session_(nullptr),
inferencing_(false){
    int device_count = 0;
    cudaGetDeviceCount(&device_count);

    int selected_id = 0;
    for (int i = 0; i < device_count; ++i) {
        cudaDeviceProp prop{};
        if (cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
            qDebug() << "CUDA Device" << i << ":" << prop.name;
            
            if (std::string(prop.name).find("NVIDIA") != std::string::npos) {
                selected_id = i;
                break;
            }
        }
    }
    qDebug() << "Selected CUDA device:" << selected_id;

    {
        char buf[8]; _snprintf_s(buf, _TRUNCATE, "%d", selected_id);
        _putenv_s("CUDA_VISIBLE_DEVICES", buf);
    }

    session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    session_options_.SetIntraOpNumThreads(1);
    session_options_.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);

    OrtCUDAProviderOptions cuda_options{};
    cuda_options.device_id = 0;
    session_options_.AppendExecutionProvider_CUDA(cuda_options);

    try {
        const wchar_t* model_w = L"D:/Repo/Yolov1/yolomodel.onnx";
        ort_session_ = new Ort::Session(ort_env_, model_w, session_options_);
        qDebug() << "Session created successfully (CUDA).";
    } catch (const Ort::Exception& e) {
        qCritical() << "CUDA session creation failed:" << e.what();
        qCritical() << "Falling back to CPU provider.";
        session_options_ = Ort::SessionOptions{};
        const wchar_t* model_w = L"D:/Repo/Yolov1/yolomodel.onnx";
        ort_session_ = new Ort::Session(ort_env_, model_w, session_options_);
    }

    for (auto& p : Ort::GetAvailableProviders())
        qDebug() << "Provider:" << QString::fromStdString(p);


    {
        Ort::AllocatorWithDefaultOptions allocator;
        Ort::AllocatedStringPtr in_ptr = ort_session_->GetInputNameAllocated(0, allocator);
        Ort::AllocatedStringPtr out_ptr = ort_session_->GetOutputNameAllocated(0, allocator);
        input_name_str_ = std::string(in_ptr.get());
        output_name_str_ = std::string(out_ptr.get());
        input_names_ = { input_name_str_.c_str() };
        output_names_ = { output_name_str_.c_str() };
        qDebug() << "IO names:" << input_names_[0] << "->" << output_names_[0];
    }

    qDebug() << "YOLOv1 ONNX model ready.";
}

Inference::~Inference() {
    delete ort_session_;
    ort_session_ = nullptr;
}

void Inference::detectObjects(cv::Mat& frame) {
	qDebug() << "inferencing";
	cv::Mat blob = cv::dnn::blobFromImage(
		frame,
		1.0 / 255.0,
		cv::Size(frame.cols, frame.rows),
		cv::Scalar(),
		true,
		false,
		CV_32F
	);
	constexpr int S = 7, B = 2, C = 20;
	constexpr int H = 448, W = 448;
	constexpr std::array<int64_t, 4> input_shape{ 1, 3, 448, 448 };

	std::vector<float> input_tensor_values(1 * 3 * H * W);

	float* blob_data = blob.ptr<float>();
	const Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
		OrtAllocatorType::OrtArenaAllocator, OrtMemTypeDefault
	);

	const Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
		memory_info,
		blob_data,
		1 * 3 * 448 * 448,
		input_shape.data(),
		input_shape.size()
	);

	auto output_tensors = ort_session_->Run(
		Ort::RunOptions{ nullptr },
		input_names_.data(), &input_tensor, 1,
		output_names_.data(), 1
	);

	float* preds = output_tensors.front().GetTensorMutableData<float>();

	constexpr float conf_thresh = 0.15f;
	constexpr int stride = (5 * B + C);

	for (int i = 0; i < S; ++i) {
		for (int j = 0; j < S; ++j) {
			const int offset = S * S;

			int   best_id = 0;
			float best_v = -std::numeric_limits<float>::infinity();

			for (int clsidx = 0; clsidx < 20; ++clsidx) {
				float v = preds[(i * S + j) + (clsidx * offset)];
				if (v > best_v) {
					best_v = v;
					best_id = clsidx;
				}
			}

			int   class_id = best_id;
			float class_conf = best_v;


			for (int b = 0; b < B; ++b) {
				float x_cell = preds[(i * S + j) + (21 * offset)];
				float y_cell = preds[(i * S + j) + (22 * offset)];
				float w = preds[(i * S + j) + (23 * offset)];
				float h = preds[(i * S + j) + (24 * offset)];
				float conf = preds[(i * S + j) + (20 * offset)];

				float score = conf * class_conf;
				if (score < conf_thresh)
					continue;

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

				std::string label = std::to_string(class_id) + ":" + cv::format("%.2f", score);
				cv::putText(frame, label, cv::Point(x1, y1 - 3),
					cv::FONT_HERSHEY_SIMPLEX, 0.3, color, 1);
			}
		}
	}
}

void Inference::runInference(cv::Mat& frame) {
	inferencing_ = true;
    detectObjects(frame);
	inferencing_ = false;
    emit InferenceDone();
}

void Inference::finishInference() {
	
}
