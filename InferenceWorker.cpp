#include "InferenceWorker.h"

InferenceWorker::InferenceWorker(QObject *parent, cv::Mat frame, QMutex* lock)
: QObject(parent),
frame_(frame),
inferLock_(lock) {
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
	//session_options_.DisableMemPattern();
	//session_options_.DisableCpuMemArena();

    OrtCUDAProviderOptions cuda_options{};
    cuda_options.device_id = 0;
	//cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
    session_options_.AppendExecutionProvider_CUDA(cuda_options);

    try {
        const wchar_t* model_w = L"C:/Users/sejun/source/repos/CVS/yolomodel.onnx";
        ort_session_ = new Ort::Session(ort_env_, model_w, session_options_);
        qDebug() << "Session created successfully (CUDA).";
    }
    catch (const Ort::Exception& e) {
        qCritical() << "CUDA session creation failed:" << e.what();
        qCritical() << "Falling back to CPU provider.";
        session_options_ = Ort::SessionOptions{};
        const wchar_t* model_w = L"C:/Users/sejun/source/repos/CVS/yolomodel.onnx";
        ort_session_ = new Ort::Session(ort_env_, model_w, session_options_);
    }

    for (auto& p : Ort::GetAvailableProviders())
        qDebug() << "Provider:" << QString::fromStdString(p);

    
    Ort::AllocatorWithDefaultOptions allocator;
    Ort::AllocatedStringPtr in_ptr = ort_session_->GetInputNameAllocated(0, allocator);
    Ort::AllocatedStringPtr out_ptr = ort_session_->GetOutputNameAllocated(0, allocator);
    input_name_str_ = std::string(in_ptr.get());
    output_name_str_ = std::string(out_ptr.get());
    input_names_ = { input_name_str_.c_str() };
    output_names_ = { output_name_str_.c_str() };
    qDebug() << "IO names:" << input_names_[0] << "->" << output_names_[0];
    

    qDebug() << "YOLOv1 ONNX model ready.";

	cudaMalloc((void**)&dptr_, sizeof(float) * 10);
}

InferenceWorker::~InferenceWorker() {
    delete ort_session_;
    ort_session_ = nullptr;
}

void InferenceWorker::run() {
	qDebug() << "inferencing";

	//testCudaKernel();

	QElapsedTimer t_all;
	t_all.start();

	QElapsedTimer t_blob;
	t_blob.start();
	inferLock_->lock();
	cv::Mat blob = cv::dnn::blobFromImage(
		frame_,
		1.0 / 255.0,
		cv::Size(448, 448),
		cv::Scalar(),
		false,
		true,
		CV_32F
	);	
	inferLock_->unlock();
	qint64 ns_blob = t_blob.nsecsElapsed();
	qDebug() << "[blob] latency =" << ns_blob / 1e03 << "us";

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

	QElapsedTimer t;
	t.start();
	auto output_tensors = ort_session_->Run(
		Ort::RunOptions{ nullptr },
		input_names_.data(), &input_tensor, 1,
		output_names_.data(), 1
	);
	qint64 ns = t.nsecsElapsed();

	qDebug() << "[ORT Run] latency =" << ns / 1e03 << "us";


	QElapsedTimer post_t;
	post_t.start();
	float* preds = output_tensors.front().GetTensorMutableData<float>();

	std::vector<cv::Rect> boxes;
	std::vector<float> scores;
	std::vector<int> nms_indices;
	std::vector<int> cls_indices;
	std::vector<std::string> classes = { "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor" };

	constexpr float score_thresh = 0.1f;
	constexpr float nms_thresh = 0.3f;

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

			float boxScore1 = class_conf * preds[(i * S + j) + (20 * offset)];
			float boxScore2 = class_conf * preds[(i * S + j) + (25 * offset)];

			float x1, y1, w1, h1;
			x1 = ((preds[(i * S + j) + (21 * offset)] + j) / S) * 448;
			y1 = ((preds[(i * S + j) + (22 * offset)] + i) / S) * 448;
			w1 = preds[(i * S + j) + (23 * offset)] * 448;
			h1 = preds[(i * S + j) + (24 * offset)] * 448;

			float x2, y2, w2, h2;
			x2 = ((preds[(i * S + j) + (26 * offset)] + j) / S) * 448;
			y2 = ((preds[(i * S + j) + (27 * offset)] + i) / S) * 448;
			w2 = preds[(i * S + j) + (28 * offset)] * 448;
			h2 = preds[(i * S + j) + (29 * offset)] * 448;

			cv::Rect2f box1(
				x1 - (w1 / 2.0f),
				y1 - (h1 / 2.0f),
				w1,
				h1
			);

			cv::Rect2f box2(
				x2 - (w2 / 2.0f),
				y2 - (h2 / 2.0f),
				w2,
				h2
			);

			boxes.push_back(box1);
			boxes.push_back(box2);
			scores.push_back(boxScore1);
			scores.push_back(boxScore2);
			cls_indices.push_back(class_id);
			cls_indices.push_back(class_id);
		}
	}

	cv::dnn::NMSBoxes(
		boxes,
		scores,
		score_thresh,
		nms_thresh,
		nms_indices
	);

	inferLock_->lock();
	for (auto cls : nms_indices) {
		cv::rectangle(
			frame_,
			boxes[cls],
			cv::Scalar(0, 255, 0),
			3
		);

		std::string text = classes[cls_indices[cls]];

		int baseline = 0;
		cv::Size textSize = cv::getTextSize(
			text,
			cv::FONT_HERSHEY_SIMPLEX,
			0.6,
			2,
			&baseline
		);

		int textY = boxes[cls].y - 2;
		if (textY < textSize.height)
			textY = boxes[cls].y + textSize.height + 2;

		cv::Point org(boxes[cls].x, textY);

		cv::putText(frame_, text, org,
			cv::FONT_HERSHEY_SIMPLEX,
			0.6,
			cv::Scalar(0, 255, 0),
			2);
	}
	inferLock_->unlock();
	qint64 ns_post = post_t.nsecsElapsed();
	qDebug() << "[Postprocess] latency =" << ns_post / 1e03 << "us";

	qint64 ns_all = t_all.nsecsElapsed();
	qDebug() << "[All] latency =" << ns_all / 1e03 << "us";
}

void InferenceWorker::testCudaKernel() {
	const int N = 10;
	float h_data[N];

	for (int i = 0; i < N; i++) h_data[i] = i;

	float* d_data;
	cudaMalloc((void**)&d_data, sizeof(float) * N);
	cudaMemcpy(d_data, h_data, sizeof(float) * N, cudaMemcpyHostToDevice);

	launchMyKernel(d_data, N, 0);
	cudaDeviceSynchronize();

	cudaMemcpy(h_data, d_data, sizeof(float) * N, cudaMemcpyDeviceToHost);

	for (int i = 0; i < N; i++) {
		qDebug() << h_data[i];
	}

	cudaFree(d_data);
}