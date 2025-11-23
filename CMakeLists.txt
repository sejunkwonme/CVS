cmake_minimum_required(VERSION 3.31)
project(CVS VERSION 0.1 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(OpenCV_DIR "D:/Library/opencv4110_debug/x64/vc17/staticlib")
set(GSTREAMER_DIR "D:/Library/gstreamer_devel/1.0/msvc_x86_64")
add_definitions(-DOpenCV_STATIC)


find_package(QT NAMES Qt6 Qt5 REQUIRED COMPONENTS Widgets)
find_package(Qt${QT_VERSION_MAJOR}
REQUIRED COMPONENTS Widgets Multimedia Concurrent)
find_package(OpenCV REQUIRED)
find_package(TBB REQUIRED)
find_package(CUDAToolkit REQUIRED)
#find_package(yaml-cpp REQUIRED)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)


qt_add_executable(${PROJECT_NAME}
    main.cpp
    MainWindow.h MainWindow.cpp
    Utilities.h Utilities.cpp
    CaptureController.h CaptureController.cpp
    CaptureWorker.h CaptureWorker.cpp
    InferenceController.h InferenceController.cpp
    InferenceWorker.h InferenceWorker.cpp
    FrameController.h FrameController.cpp
    FrameWorker.h FrameWorker.cpp
)

target_link_libraries(${PROJECT_NAME} PRIVATE
    Qt${QT_VERSION_MAJOR}::Widgets
    Qt${QT_VERSION_MAJOR}::Multimedia
    Qt${QT_VERSION_MAJOR}::Concurrent
    ${OpenCV_LIBS}
    TBB::tbb
    onnxruntime
    CUDA::cudart
    gstapp-1.0
    gstreamer-1.0
    gstbase-1.0
    gobject-2.0
    glib-2.0
    #yaml-cpp
)

target_include_directories(${PROJECT_NAME} PRIVATE
    "D:/Library/onnxruntime-win-x64-gpu-1.23.2/include"
    ${GSTREAMER_DIR}/include/gstreamer-1.0
    ${GSTREAMER_DIR}/include/glib-2.0
    ${GSTREAMER_DIR}/lib/glib-2.0/include
    ${OpenCV_INCLUDE_DIRS}
    #"D:/Library/yaml-cpp-debug/include/yaml-cpp"

)

target_link_directories(${PROJECT_NAME} PRIVATE
    "C:/Program Files (x86)/Intel/oneAPI/tbb/2022.3/lib"
    "C:/Program Files (x86)/Intel/oneAPI/ipp/2022.3/lib"
    "D:/Library/onnxruntime-win-x64-gpu-1.23.2/lib"
    ${GSTREAMER_DIR}/lib
    ${GSTREAMER_DIR}/lib/gstreamer-1.0
    #"D:/Library/yaml-cpp-debug/lib"
)
