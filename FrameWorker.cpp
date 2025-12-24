#include "FrameWorker.h"

FrameWorker::FrameWorker(QObject* parent, CaptureController* capC, InferenceController* inferC, cv::Mat(&frame)[4], unsigned char* gui_image)
: QObject(parent),
capC_(capC),
inferC_(inferC) {
    frame_[0] = frame[0];
    frame_[1] = frame[1];
	frame_[2] = frame[2];
	frame_[3] = frame[3];
    gui_image_ = gui_image;
}

FrameWorker::~FrameWorker() {

}

void FrameWorker::writeBuffer(uint64_t framecount) {
    //const int cur = static_cast<int>(framecount & 1); // 현재 버퍼 인덱스
    //const int prev = static_cast<int>((framecount + 1) & 1); //과거 버퍼 인덱스 (단순히 반대)

	// 현재 프레임이 써야 할 슬롯
	int cur = framecount % 4;
    // 항상 현재에 쓴다
    cudaMemcpy(frame_[cur].data, gui_image_, sizeof(unsigned char) * 1 * 3 * 448 * 448, cudaMemcpyDeviceToHost);
}

// render 할 때는 항상 과거의 것을 가져와서 box그린다음 gui에 렌더링한다
// 그런데 inferenceworker_post 에서 온것은 과거의 인덱스 그대로이므로 이걸 그대로 사용
void FrameWorker::renderImage(float* d_out, uint64_t framecount) {
	//const int cur = static_cast<int>(framecount & 1); // 현재 버퍼 인덱스
	//const int prev = static_cast<int>((framecount + 1) & 1); //과거 버퍼 인덱스 (단순히 반대)
	int cur = framecount % 4;
	constexpr int S = 7, B = 2, C = 20;
	constexpr int H = 448, W = 448;
	float preds[1470];
	cudaMemcpy(preds, d_out, sizeof(float) * 1470, cudaMemcpyDeviceToHost);
	std::vector<std::vector<cv::Rect>> boxes(20);
	std::vector<std::vector<float>> score(20);
	std::vector<std::vector<int>> indices(20);
	std::vector<std::string> classes = { "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor" };
	cv::Mat scoreMatrix(20, 98, CV_32F, cv::Scalar(0));
	constexpr float score_thresh = 0.2f;
	constexpr float nms_thresh = 0.5f;
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

	for (int boxidx = 0; boxidx < S * S * 2 - 1; boxidx++) {
		double maxScore;
		int maxindex[2];
		cv::minMaxIdx(scoreMatrix.col(boxidx), nullptr, &maxScore, nullptr, maxindex);
		if (maxScore > 0.0) {
			cv::rectangle(
				frame_[cur],
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
			cv::putText(frame_[cur], text, org,
				cv::FONT_HERSHEY_SIMPLEX,
				0.6,
				cv::Scalar(0, 255, 0),
				2);
		}
	}
    emit renderCompleted(frame_[cur]);
}