cmake_minimum_required(VERSION 3.31)
project(CVS VERSION 0.1 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(PkgConfig REQUIRED)
find_package(Qt6 REQUIRED COMPONENTS Widgets Multimedia Concurrent HINTS "D:Library/Qt/6.8.3/msvc2022_64")
find_package(OpenCV REQUIRED HINTS "D:/Library/opencv4110_debug/x64/vc17/staticlib")
find_package(TBB REQUIRED HINTS "C:/Program Files (x86)/Intel/oneAPI/tbb/2022.3/lib/cmake/tbb")
find_package(CUDAToolkit REQUIRED)

pkg_check_modules(GST REQUIRED
    gstreamer-1.0
    gstreamer-app-1.0
    gstreamer-video-1.0
    gstreamer-base-1.0
)
add_definitions(-DOpenCV_STATIC)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)

add_library(CVS_CUDA STATIC
    kernel.cu
)
target_compile_options(CVS_CUDA PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
)

set_source_files_properties(kernel.cu PROPERTIES LANGUAGE CUDA)

target_link_libraries(CVS_CUDA PUBLIC CUDA::cudart)

target_include_directories(CVS_CUDA PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
)
set_target_properties(CVS_CUDA PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_ARCHITECTURES 61 75 86  # RTX 3090
)

#set(CMAKE_CUDA_FLAGS_DEBUG "-G -lineinfo -Xcompiler=\"/Zi /Od\"")
set(ORT_ROOT "D:/Library/onnxruntime-win-x64-gpu-1.23.2")

# Core runtime
add_library(onnxruntime SHARED IMPORTED)
set_target_properties(onnxruntime PROPERTIES
    IMPORTED_LOCATION             "${ORT_ROOT}/lib/onnxruntime.dll"
    IMPORTED_IMPLIB               "${ORT_ROOT}/lib/onnxruntime.lib"
    INTERFACE_INCLUDE_DIRECTORIES "${ORT_ROOT}/include"
)

# Shared provider
add_library(onnxruntime_providers_shared SHARED IMPORTED)
set_target_properties(onnxruntime_providers_shared PROPERTIES
    IMPORTED_LOCATION "${ORT_ROOT}/lib/onnxruntime_providers_shared.dll"
    IMPORTED_IMPLIB   "${ORT_ROOT}/lib/onnxruntime_providers_shared.lib"
)

# CUDA provider
add_library(onnxruntime_providers_cuda SHARED IMPORTED)
set_target_properties(onnxruntime_providers_cuda PROPERTIES
    IMPORTED_LOCATION "${ORT_ROOT}/lib/onnxruntime_providers_cuda.dll"
    IMPORTED_IMPLIB   "${ORT_ROOT}/lib/onnxruntime_providers_cuda.lib"
)

# TensorRT provider (optional)
add_library(onnxruntime_providers_tensorrt SHARED IMPORTED)
set_target_properties(onnxruntime_providers_tensorrt PROPERTIES
    IMPORTED_LOCATION "${ORT_ROOT}/lib/onnxruntime_providers_tensorrt.dll"
    IMPORTED_IMPLIB   "${ORT_ROOT}/lib/onnxruntime_providers_tensorrt.lib"
)

set(ORT_DLLS
    onnxruntime.dll
    onnxruntime_providers_shared.dll
    onnxruntime_providers_tensorrt.dll
    onnxruntime_providers_cuda.dll
)

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
    Qt6::Widgets
    Qt6::Multimedia
    Qt6::Concurrent
    ${OpenCV_LIBS}
    TBB::tbb
    onnxruntime
    onnxruntime_providers_shared
    onnxruntime_providers_cuda
    onnxruntime_providers_tensorrt
    ${GST_LIBRARIES}
    CVS_CUDA
)
target_include_directories(${PROJECT_NAME} PRIVATE
    "D:/Library/onnxruntime-win-x64-gpu-1.23.2/include"
    ${GST_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)
target_link_directories(${PROJECT_NAME} PRIVATE
    ${GST_LIBRARY_DIRS}
    "C:/Program Files (x86)/Intel/oneAPI/tbb/2022.3/lib"
    "C:/Program Files (x86)/Intel/oneAPI/ipp/2022.3/lib"
    "D:/Library/onnxruntime-win-x64-gpu-1.23.2/lib"
)

target_compile_options(${PROJECT_NAME} PRIVATE
    $<$<CONFIG:Debug>:/Zi /Od>
)
foreach(dll ${ORT_DLLS})
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${ORT_ROOT}/lib/${dll}"
            $<TARGET_FILE_DIR:${PROJECT_NAME}>
    )
endforeach()