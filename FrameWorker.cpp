#include "FrameWorker.h"

FrameWorker::FrameWorker(QObject* parent, CaptureController* capC, InferenceController* inferC, cv::Mat(&frame)[2], unsigned char* gui_image)
: QObject(parent),
capC_(capC),
inferC_(inferC) {
    frame_[0] = frame[0];
    frame_[1] = frame[1];
    gui_image_ = gui_image;
}

FrameWorker::~FrameWorker() {

}

void FrameWorker::writeBuffer(uint64_t framecount) {
    const int cur = static_cast<int>(framecount & 1); // 현재 버퍼 인덱스
    const int prev = static_cast<int>((framecount + 1) & 1); //과거 버퍼 인덱스 (단순히 반대)

    // 항상 현재에 쓴다
    cudaMemcpy(frame_[cur].data, gui_image_, sizeof(unsigned char) * 1 * 3 * 448 * 448, cudaMemcpyDeviceToHost);
}

// render 할 때는 항상 과거의 것을 가져와서 box그린다음 gui에 렌더링한다
// 그런데 inferenceworker_post 에서 온것은 과거의 인덱스 그대로이므로 이걸 그대로 사용
void FrameWorker::renderImage(std::vector<cv::Rect2f> boxes, std::vector<int> cls, uint64_t framecount) {
    std::vector<std::string> classes = { "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor" };

    const int cur = static_cast<int>(framecount & 1); // 현재 버퍼 인덱스
    const int prev = static_cast<int>((framecount + 1) & 1); //과거 버퍼 인덱스 (단순히 반대)

    
    for (auto box : boxes) {
        cv::rectangle(
            frame_[cur],
            box,
            cv::Scalar(0, 255, 0),
            3
        );

        /*
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
            */
    }
    emit renderCompleted(frame_[cur]);
}