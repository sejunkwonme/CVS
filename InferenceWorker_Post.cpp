#include "InferenceWorker_Post.h"

InferenceWorker_Post::InferenceWorker_Post(QObject *parent, cv::Mat frame, QMutex* lock, float* ml_middle_image)
: QObject(parent) {
	middle_image_ = ml_middle_image;
	cudaStreamCreateWithFlags(&head_stream_, cudaStreamNonBlocking);
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

	session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);
	session_options_.SetIntraOpNumThreads(1);
	session_options_.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);


	OrtCUDAProviderOptions cuda_options{};
	cuda_options.device_id = 0;
	cuda_options.has_user_compute_stream = 1;
	cuda_options.user_compute_stream = (void*)head_stream_;
	cuda_options.do_copy_in_default_stream = 0;
	session_options_.AppendExecutionProvider_CUDA(cuda_options);

	try {
		const wchar_t* model_w = L"D:/Repo/Yolov1/thirdmodel-conv4-detection.onnx";
		ort_session_ = new Ort::Session(ort_env_, model_w, session_options_);
		qDebug() << "Session created successfully (CUDA).";
	}
	catch (const Ort::Exception& e) {
		qCritical() << "CUDA session creation failed:" << e.what();
		qCritical() << "Falling back to CPU provider.";
		session_options_ = Ort::SessionOptions{};
		const wchar_t* model_w = L"D:/Repo/Yolov1/thirdmodel-conv4-detection.onnx";
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

	qDebug() << "YOLOv1 detection head ready.";

	auto cuda_mem = Ort::MemoryInfo("Cuda", OrtAllocatorType::OrtDeviceAllocator, 0, OrtMemTypeDefault);

	// input device 텐서 미리 정의
	std::array<int64_t, 4> input_shape{ 1, 1024, 14, 14 };
	input_gpu_ = Ort::Value::CreateTensor<float>(
		cuda_mem,
		middle_image_,
		1 * 1024 * 14 * 14,
		input_shape.data(),
		input_shape.size()
	);

	// output device 텐서 미리 정의
	std::array<int64_t, 2> out_shape{ 1, 1470 };
	cudaMalloc((void**)&d_out_, 1 * 30 * 7 * 7 * sizeof(float));
	output_gpu_ = Ort::Value::CreateTensor<float>(
		cuda_mem,
		d_out_,
		1 * 30 * 7 * 7,
		out_shape.data(),
		out_shape.size()
	);
	binding_ = new Ort::IoBinding (*ort_session_);
	binding_->BindInput(input_name_str_.c_str(), input_gpu_);
	binding_->BindOutput(output_name_str_.c_str(), output_gpu_);

	run_opts_.AddConfigEntry("disable_synchronize_execution_providers", "1");
}

InferenceWorker_Post::~InferenceWorker_Post() {
}

void InferenceWorker_Post::run() {
	constexpr int S = 7, B = 2, C = 20;
	constexpr int H = 448, W = 448;

	//cudaEvent_t e0, e1;
	//cudaEventCreate(&e0);
	//cudaEventCreate(&e1);

	//cudaEventRecord(e0, head_stream_);
	ort_session_->Run(run_opts_, *binding_);
	//cudaEventRecord(e1, head_stream_);
	//cudaEventSynchronize(e1);

	//float ms = 0.f;
	//cudaEventElapsedTime(&ms, e0, e1);
	//qDebug() << "[head - runtime]" << ms << "ms";

	float preds[1470];
	//cudaMemcpy(preds, d_out_, sizeof(float) * 1470, cudaMemcpyDeviceToHost);
	/*
	std::vector<std::vector<cv::Rect>> boxes(20);
	std::vector<std::vector<float>> score(20);
	std::vector<std::vector<int>> indices(20);
	std::vector<std::string> classes = { "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor" };
	cv::Mat scoreMatrix(20, 98, CV_32F, cv::Scalar(0));

	constexpr float score_thresh = 0.2f;
	constexpr float nms_thresh = 0.6f;

	for (int cidx = 0; cidx < 20; cidx++) {
		for (int i = 0; i < S; ++i) {
			for (int j = 0; j < S; ++j) {
				const int offset = S * S;


				float boxScore1 = preds[(i * S + j) + (cidx * offset)] * preds[(i * S + j) + (20 * offset)];
				float boxScore2 = preds[(i * S + j) + (cidx * offset)] * preds[(i * S + j) + (25 * offset)];

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

				boxes[cidx].push_back(box1);
				boxes[cidx].push_back(box2);
				score[cidx].push_back(boxScore1);
				score[cidx].push_back(boxScore2);
			}
		}

		cv::dnn::NMSBoxes(
			boxes[cidx],
			score[cidx],
			score_thresh,
			nms_thresh,
			indices[cidx]
		);

		for (int c = 0; c < 20; c++) {
			for (int index : indices[c]) {
				scoreMatrix.at<float>(c, index) = score[c][index];
			}
		}
	}
	*/
	/*
	inferLock_->lock();
	for (int boxidx = 0; boxidx < S * S * 2 - 1; boxidx++) {
		double maxScore;
		int maxindex[2];
		cv::minMaxIdx(scoreMatrix.col(boxidx), nullptr, &maxScore, nullptr, maxindex);

		if (maxScore > 0.0) {
			cv::rectangle(
				frame_,
				boxes[maxindex[0]][boxidx],
				cv::Scalar(0, 255, 0),
				3
			);

			std::string text = classes[maxindex[0]];

			int baseline = 0;
			cv::Size textSize = cv::getTextSize(
				text,
				cv::FONT_HERSHEY_SIMPLEX,
				0.6,
				2,
				&baseline
			);

			int textY = boxes[maxindex[0]][boxidx].y - 2;
			if (textY < textSize.height)
				textY = boxes[maxindex[0]][boxidx].y + textSize.height + 2;

			cv::Point org(boxes[maxindex[0]][boxidx].x, textY);

			cv::putText(frame_, text, org,
				cv::FONT_HERSHEY_SIMPLEX,
				0.6,
				cv::Scalar(0, 255, 0),
				2);
		}
	}
	inferLock_->unlock();
	*/
}