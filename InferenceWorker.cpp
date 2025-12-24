#include "InferenceWorker.h"

InferenceWorker::InferenceWorker(QObject *parent, float** ml_image_addr, float* ml_middle_image)
: QObject(parent),
ml_image_addr_(ml_image_addr),
ml_middle_image_(ml_middle_image) {
    cudaStreamCreateWithFlags(&backboneStream_, cudaStreamNonBlocking);
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

    session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    //session_options_.SetIntraOpNumThreads(1);
    //session_options_.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);

    OrtCUDAProviderOptions cuda_options{};

    cuda_options.device_id = 0;
    cuda_options.has_user_compute_stream = 1;
    cuda_options.user_compute_stream = backboneStream_;
    cuda_options.do_copy_in_default_stream = 0;
    //cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchHeuristic;
    session_options_.AppendExecutionProvider_CUDA(cuda_options);

    try {
        const wchar_t* model_w = L"D:/Repo/Yolov1/thirdmodel-backbone.sim.onnx";
        ort_session_ = new Ort::Session(ort_env_, model_w, session_options_);
        qDebug() << "Session created successfully (CUDA).";
    } catch (const Ort::Exception& e) {
        qCritical() << "CUDA session creation failed:" << e.what();
        qCritical() << "Falling back to CPU provider.";
        session_options_ = Ort::SessionOptions{};
        const wchar_t* model_w = L"D:/Repo/Yolov1/thirdmodel-backbone.sim.onnx";
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
    
    qDebug() << "YOLOv1 backbone ready.";
	auto cuda_mem = Ort::MemoryInfo("Cuda", OrtAllocatorType::OrtDeviceAllocator, 0, OrtMemTypeDefault);

	// input device 텐서 미리 정의
	std::array<int64_t, 4> input_shape{ 1, 3, 448, 448 };
    size_t in_elem = 1 * 3 * 448 * 448;
	input_gpu_ = Ort::Value::CreateTensor<float>(
		cuda_mem,
		*ml_image_addr_,
        in_elem,
		input_shape.data(),
		input_shape.size()
	);

	// output device 텐서 미리 정의
	std::array<int64_t, 4> out_shape{ 1, 1024, 14, 14 };
    size_t out_elem = 1 * 1024 * 14 * 14;
	output_gpu_ = Ort::Value::CreateTensor<float>(
		cuda_mem,
        ml_middle_image_,
        out_elem,
		out_shape.data(),
		out_shape.size()
	);

    binding_ = new Ort::IoBinding(*ort_session_);
    binding_->BindInput(input_name_str_.c_str(), input_gpu_);
    binding_->BindOutput(output_name_str_.c_str(), output_gpu_);
}

InferenceWorker::~InferenceWorker() {
    delete ort_session_;
    ort_session_ = nullptr;

    if (backboneStream_) {
        cudaStreamDestroy(backboneStream_);
        backboneStream_ = nullptr;
    }
}

void InferenceWorker::run(uint64_t framecount) {
    //binding_->SynchronizeInputs();
	ort_session_->Run(Ort::RunOptions{nullptr}, *binding_);
    //cudaEventRecord(backboneEvent_, backboneStream_);
    //binding_->SynchronizeOutputs();
    emit backboneReady(framecount);
}