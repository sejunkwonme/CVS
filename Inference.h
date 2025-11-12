#ifndef INFERENCE_H
#define INFERENCE_H

#include <QString>
#include <QThread>
#include <QMutex>
#include <onnxruntime_cxx_api.h>
#include <cstdlib> 
#include <cuda_runtime_api.h> 

#include "opencv2/opencv.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/dnn.hpp"

class Inference : public QObject {
    Q_OBJECT
public:
    Inference();
    ~Inference();

public slots:
    void dowork() {
        for (int i = 0; i < 100000; i++) {
	        std::cout << "do";
        }
        emit finished();
    }

signals:
    void finished();
};


#endif // INFERENCE_H
