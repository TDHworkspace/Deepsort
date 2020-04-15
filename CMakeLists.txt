#cmake file for project association                     #表示注释 
#author:>---Harry
#created:>--2020/3/19
 
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)    #cmake 最低版本要求，低于2.8 构建过程会被终止。
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=1)

# Enable C++11
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE) 
 
project(deepsort)                          #定义工程名称

include_directories(
 		/usr/local/include
		/usr/local/include/opencv4
		/usr/local/include/opencv4/opencv2
        /home/xiaoqiang/Harry/setup/tensorflow-r1.4
        /home/xiaoqiang/Harry/setup/tensorflow-r1.4/tensorflow/contrib/makefile/downloads/eigen
        /home/xiaoqiang/Harry/setup/tensorflow-r1.4/tensorflow/contrib/makefile/gen/protobuf/include
        /home/xiaoqiang/Harry/setup/tensorflow-r1.4/tensorflow/contrib/makefile/downloads/nsync/public
		/home/xiaoqiang/Harry/setup/tensorflow-r1.4/bazel-genfiles
		/home/xiaoqiang/Harry/setup/tensorflow-r1.4/bazel-bin/tensorflow		   	
)

set(SC_FILES "./DeepAppearanceDescriptor/FeatureTensor.cpp"
	"./DeepAppearanceDescriptor/FeatureTensor.h"
	"./DeepAppearanceDescriptor/model.cpp"
	"DeepAppearanceDescriptor/model.h"
	"./KalmanFilter/kalmanfilter.cpp"
	"./KalmanFilter/kalmanfilter.h"
	"./KalmanFilter/linear_assignment.cpp"
	"./KalmanFilter/linear_assignment.h"
	"./KalmanFilter/nn_matching.cpp"
	"./KalmanFilter/nn_matching.h"
	"./KalmanFilter/track.cpp"
	"./KalmanFilter/track.h"
	"./KalmanFilter/tracker.cpp"
	"./KalmanFilter/tracker.h"
	"./MunkresAssignment/munkres/munkres.cpp"
	"./MunkresAssignment/munkres/munkres.h"
	"./MunkresAssignment/hungarianoper.cpp"
	"./MunkresAssignment/hungarianoper.h"
	"./main.cpp"

)
 
# Declare the executable target built from your sources
add_executable(${PROJECT_NAME} ${SC_FILES} )

# Link your application with OpenCV libraries
target_link_libraries (${PROJECT_NAME} /usr/local/lib/libopencv_calib3d.so)
target_link_libraries (${PROJECT_NAME} /usr/local/lib/libopencv_core.so)
target_link_libraries (${PROJECT_NAME} /usr/local/lib/libopencv_highgui.so)
target_link_libraries (${PROJECT_NAME} /usr/local/lib/libopencv_imgproc.so)
target_link_libraries (${PROJECT_NAME} /usr/local/lib/libopencv_imgcodecs.so)
target_link_libraries (${PROJECT_NAME} /usr/local/lib/libopencv_objdetect.so)
target_link_libraries (${PROJECT_NAME} /usr/local/lib/libopencv_photo.so)
target_link_libraries (${PROJECT_NAME} /usr/local/lib/libopencv_dnn.so)
target_link_libraries (${PROJECT_NAME} /usr/local/lib/libopencv_features2d.so)
target_link_libraries (${PROJECT_NAME} /usr/local/lib/libopencv_stitching.so)
target_link_libraries (${PROJECT_NAME} /usr/local/lib/libopencv_flann.so)
target_link_libraries (${PROJECT_NAME} /usr/local/lib/libopencv_videoio.so)
target_link_libraries (${PROJECT_NAME} /usr/local/lib/libopencv_video.so)
target_link_libraries (${PROJECT_NAME} /usr/local/lib/libopencv_ml.so)


# Link your application with tensorflow libraries
target_link_libraries (${PROJECT_NAME} /home/xiaoqiang/Harry/setup/tensorflow-r1.4/bazel-bin/tensorflow/libtensorflow_cc.so)
target_link_libraries (${PROJECT_NAME} /home/xiaoqiang/Harry/setup/tensorflow-r1.4/bazel-bin/tensorflow/libtensorflow_framework.so)     

