# Deep-SORT
MOT using deepsort yolo3 with C++

操作系统：Ubuntu 18.04

深度学习的模型分两个，一个是目标检测，一个是目标跟踪

目标检测的模型
https://pjreddie.com/darknet/yolo/
用的是OpenCV加载的，所以更换成您想要使用的模型

目标跟踪中特征部分 
目标跟踪模型 mars-small128 

OpenCV的DNN加载YOLO模型，这样就不用依赖Darknet库
不依赖cuda，cudnn，这样方便环境搭建
现在目标跟踪的特征部分使用TensorFlow C++的api。如果再想轻量级一些，就要去除Tensorflow的依赖。

环境配置可以参考：https://blog.csdn.net/tdh2017/article/details/105139097
代码调试参考：https://blog.csdn.net/tdh2017/article/details/105527691

里面使用了github作者的大量代码，站在巨人们的基础上。


如果您要使用我的代码搭建环境，您要做的是
１．按照我上面那个网站把所需软件安装好
２．修改　CMakeLists.txt　　
    主要是头文件，库文件的路径更改成您自己的文件所在路径
３．cd build/　
    make ..
    注：模型文件放置与生成文件相同的目录　https://pan.baidu.com/s/1aS7N9ZVffYrMjDafwuisGg 提取码: u6pv
　　　　./deepsort --video=run.mp4


本实例代码已编译通过，且正常运行。

2020-04-14 16:44:45.624974: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1030] Found device 0 with properties: 
name: GeForce GTX 1080 major: 6 minor: 1 memoryClockRate(GHz): 1.7715
pciBusID: 0000:01:00.0
totalMemory: 7.92GiB freeMemory: 7.50GiB
2020-04-14 16:44:45.625063: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1120] Creating TensorFlow device (/device:GPU:0) -> (device: 0, name: GeForce GTX 1080, pci bus id: 0000:01:00.0, compute capability: 6.1)
node name:Placeholder
node name:Cast
node name:map/Shape
node name:map/strided_slice/stack
node name:map/strided_slice/stack_1
node name:map/strided_slice/stack_2
node name:map/strided_slice
node name:map/TensorArray
node name:map/TensorArrayUnstack/Shape
node name:map/TensorArrayUnstack/strided_slice/stack
node name:map/TensorArrayUnstack/strided_slice/stack_1
node name:map/TensorArrayUnstack/strided_slice/stack_2
node name:map/TensorArrayUnstack/strided_slice
node name:map/TensorArrayUnstack/range/start
node name:map/TensorArrayUnstack/range/delta
node name:map/TensorArrayUnstack/range
node name:map/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3
node name:map/Const
node name:map/TensorArray_1
node name:map/while/Enter
node name:map/while/Enter_1
node name:map/while/Merge
node name:map/while/Merge_1
node name:map/while/Less/Enter
node name:map/while/Less
node name:map/while/LoopCond
node name:map/while/Switch
node name:map/while/Switch_1
node name:map/while/Identity
node name:map/while/Identity_1
node name:map/while/TensorArrayReadV3/Enter
node name:map/while/TensorArrayReadV3/Enter_1
node name:map/while/TensorArrayReadV3
node name:map/while/strided_slice/stack
node name:map/while/strided_slice/stack_1
node name:map/while/strided_slice/stack_2
node name:map/while/strided_slice
node name:map/while/TensorArrayWrite/TensorArrayWriteV3/Enter
node name:map/while/TensorArrayWrite/TensorArrayWriteV3
node name:map/while/add/y
node name:map/while/add
node name:map/while/NextIteration
node name:map/while/NextIteration_1
node name:map/while/Exit
node name:map/while/Exit_1
node name:map/TensorArrayStack/TensorArraySizeV3
node name:map/TensorArrayStack/range/start
node name:map/TensorArrayStack/range/delta
node name:map/TensorArrayStack/range
node name:map/TensorArrayStack/TensorArrayGatherV3
node name:conv1_1/weights/Initializer/truncated_normal/shape
node name:conv1_1/weights/Initializer/truncated_normal/mean
node name:conv1_1/weights/Initializer/truncated_normal/stddev
node name:conv1_1/weights/Initializer/truncated_normal/TruncatedNormal
node name:conv1_1/weights/Initializer/truncated_normal/mul
node name:conv1_1/weights/Initializer/truncated_normal
node name:conv1_1/weights
node name:conv1_1/weights/Assign
node name:conv1_1/weights/read
node name:conv1_1/kernel/Regularizer/l2_regularizer/scale
node name:conv1_1/kernel/Regularizer/l2_regularizer/L2Loss
node name:conv1_1/kernel/Regularizer/l2_regularizer
node name:conv1_1/convolution/Shape
node name:conv1_1/convolution/dilation_rate
node name:conv1_1/convolution
node name:conv1_1/conv1_1/bn/beta/Initializer/zeros
node name:conv1_1/conv1_1/bn/beta
node name:conv1_1/conv1_1/bn/beta/Assign
node name:conv1_1/conv1_1/bn/beta/read
node name:conv1_1/conv1_1/bn/moving_mean/Initializer/zeros
node name:conv1_1/conv1_1/bn/moving_mean
node name:conv1_1/conv1_1/bn/moving_mean/Assign
node name:conv1_1/conv1_1/bn/moving_mean/read
node name:conv1_1/conv1_1/bn/moving_variance/Initializer/ones
node name:conv1_1/conv1_1/bn/moving_variance
node name:conv1_1/conv1_1/bn/moving_variance/Assign
node name:conv1_1/conv1_1/bn/moving_variance/read
node name:conv1_1/conv1_1/bn/batchnorm/add/y
node name:conv1_1/conv1_1/bn/batchnorm/add
node name:conv1_1/conv1_1/bn/batchnorm/Rsqrt
node name:conv1_1/conv1_1/bn/batchnorm/mul
node name:conv1_1/conv1_1/bn/batchnorm/mul_1
node name:conv1_1/conv1_1/bn/batchnorm/sub
node name:conv1_1/conv1_1/bn/batchnorm/add_1
node name:conv1_1/Elu
node name:conv1_2/weights/Initializer/truncated_normal/shape
node name:conv1_2/weights/Initializer/truncated_normal/mean
node name:conv1_2/weights/Initializer/truncated_normal/stddev
node name:conv1_2/weights/Initializer/truncated_normal/TruncatedNormal
node name:conv1_2/weights/Initializer/truncated_normal/mul
node name:conv1_2/weights/Initializer/truncated_normal
node name:conv1_2/weights
node name:conv1_2/weights/Assign
node name:conv1_2/weights/read
node name:conv1_2/kernel/Regularizer/l2_regularizer/scale
node name:conv1_2/kernel/Regularizer/l2_regularizer/L2Loss
node name:conv1_2/kernel/Regularizer/l2_regularizer
node name:conv1_2/convolution/Shape
node name:conv1_2/convolution/dilation_rate
node name:conv1_2/convolution
node name:conv1_2/conv1_2/bn/beta/Initializer/zeros
node name:conv1_2/conv1_2/bn/beta
node name:conv1_2/conv1_2/bn/beta/Assign
node name:conv1_2/conv1_2/bn/beta/read
node name:conv1_2/conv1_2/bn/moving_mean/Initializer/zeros
node name:conv1_2/conv1_2/bn/moving_mean
node name:conv1_2/conv1_2/bn/moving_mean/Assign
node name:conv1_2/conv1_2/bn/moving_mean/read
node name:conv1_2/conv1_2/bn/moving_variance/Initializer/ones
node name:conv1_2/conv1_2/bn/moving_variance
node name:conv1_2/conv1_2/bn/moving_variance/Assign
node name:conv1_2/conv1_2/bn/moving_variance/read
node name:conv1_2/conv1_2/bn/batchnorm/add/y
node name:conv1_2/conv1_2/bn/batchnorm/add
node name:conv1_2/conv1_2/bn/batchnorm/Rsqrt
node name:conv1_2/conv1_2/bn/batchnorm/mul
node name:conv1_2/conv1_2/bn/batchnorm/mul_1
node name:conv1_2/conv1_2/bn/batchnorm/sub
node name:conv1_2/conv1_2/bn/batchnorm/add_1
node name:conv1_2/Elu
node name:pool1/MaxPool
node name:conv2_1/1/weights/Initializer/truncated_normal/shape
node name:conv2_1/1/weights/Initializer/truncated_normal/mean
node name:conv2_1/1/weights/Initializer/truncated_normal/stddev
node name:conv2_1/1/weights/Initializer/truncated_normal/TruncatedNormal
node name:conv2_1/1/weights/Initializer/truncated_normal/mul
node name:conv2_1/1/weights/Initializer/truncated_normal
node name:conv2_1/1/weights
node name:conv2_1/1/weights/Assign
node name:conv2_1/1/weights/read
node name:conv2_1/1/kernel/Regularizer/l2_regularizer/scale
node name:conv2_1/1/kernel/Regularizer/l2_regularizer/L2Loss
node name:conv2_1/1/kernel/Regularizer/l2_regularizer
node name:conv2_1/1/convolution/Shape
node name:conv2_1/1/convolution/dilation_rate
node name:conv2_1/1/convolution
node name:conv2_1/1/conv2_1/1/bn/beta/Initializer/zeros
node name:conv2_1/1/conv2_1/1/bn/beta
node name:conv2_1/1/conv2_1/1/bn/beta/Assign
node name:conv2_1/1/conv2_1/1/bn/beta/read
node name:conv2_1/1/conv2_1/1/bn/moving_mean/Initializer/zeros
node name:conv2_1/1/conv2_1/1/bn/moving_mean
node name:conv2_1/1/conv2_1/1/bn/moving_mean/Assign
node name:conv2_1/1/conv2_1/1/bn/moving_mean/read
node name:conv2_1/1/conv2_1/1/bn/moving_variance/Initializer/ones
node name:conv2_1/1/conv2_1/1/bn/moving_variance
node name:conv2_1/1/conv2_1/1/bn/moving_variance/Assign
node name:conv2_1/1/conv2_1/1/bn/moving_variance/read
node name:conv2_1/1/conv2_1/1/bn/batchnorm/add/y
node name:conv2_1/1/conv2_1/1/bn/batchnorm/add
node name:conv2_1/1/conv2_1/1/bn/batchnorm/Rsqrt
node name:conv2_1/1/conv2_1/1/bn/batchnorm/mul
node name:conv2_1/1/conv2_1/1/bn/batchnorm/mul_1
node name:conv2_1/1/conv2_1/1/bn/batchnorm/sub
node name:conv2_1/1/conv2_1/1/bn/batchnorm/add_1
node name:conv2_1/1/Elu
node name:Dropout/Identity
node name:conv2_1/2/weights/Initializer/truncated_normal/shape
node name:conv2_1/2/weights/Initializer/truncated_normal/mean
node name:conv2_1/2/weights/Initializer/truncated_normal/stddev
node name:conv2_1/2/weights/Initializer/truncated_normal/TruncatedNormal
node name:conv2_1/2/weights/Initializer/truncated_normal/mul
node name:conv2_1/2/weights/Initializer/truncated_normal
node name:conv2_1/2/weights
node name:conv2_1/2/weights/Assign
node name:conv2_1/2/weights/read
node name:conv2_1/2/kernel/Regularizer/l2_regularizer/scale
node name:conv2_1/2/kernel/Regularizer/l2_regularizer/L2Loss
node name:conv2_1/2/kernel/Regularizer/l2_regularizer
node name:conv2_1/2/biases/Initializer/zeros
node name:conv2_1/2/biases
node name:conv2_1/2/biases/Assign
node name:conv2_1/2/biases/read
node name:conv2_1/2/convolution/Shape
node name:conv2_1/2/convolution/dilation_rate
node name:conv2_1/2/convolution
node name:conv2_1/2/BiasAdd
node name:add
node name:conv2_3/bn/beta/Initializer/zeros
node name:conv2_3/bn/beta
node name:conv2_3/bn/beta/Assign
node name:conv2_3/bn/beta/read
node name:conv2_3/bn/moving_mean/Initializer/zeros
node name:conv2_3/bn/moving_mean
node name:conv2_3/bn/moving_mean/Assign
node name:conv2_3/bn/moving_mean/read
node name:conv2_3/bn/moving_variance/Initializer/ones
node name:conv2_3/bn/moving_variance
node name:conv2_3/bn/moving_variance/Assign
node name:conv2_3/bn/moving_variance/read
node name:conv2_3/bn/batchnorm/add/y
node name:conv2_3/bn/batchnorm/add
node name:conv2_3/bn/batchnorm/Rsqrt
node name:conv2_3/bn/batchnorm/mul
node name:conv2_3/bn/batchnorm/mul_1
node name:conv2_3/bn/batchnorm/sub
node name:conv2_3/bn/batchnorm/add_1
node name:Elu
node name:conv2_3/1/weights/Initializer/truncated_normal/shape
node name:conv2_3/1/weights/Initializer/truncated_normal/mean
node name:conv2_3/1/weights/Initializer/truncated_normal/stddev
node name:conv2_3/1/weights/Initializer/truncated_normal/TruncatedNormal
node name:conv2_3/1/weights/Initializer/truncated_normal/mul
node name:conv2_3/1/weights/Initializer/truncated_normal
node name:conv2_3/1/weights
node name:conv2_3/1/weights/Assign
node name:conv2_3/1/weights/read
node name:conv2_3/1/kernel/Regularizer/l2_regularizer/scale
node name:conv2_3/1/kernel/Regularizer/l2_regularizer/L2Loss
node name:conv2_3/1/kernel/Regularizer/l2_regularizer
node name:conv2_3/1/convolution/Shape
node name:conv2_3/1/convolution/dilation_rate
node name:conv2_3/1/convolution
node name:conv2_3/1/conv2_3/1/bn/beta/Initializer/zeros
node name:conv2_3/1/conv2_3/1/bn/beta
node name:conv2_3/1/conv2_3/1/bn/beta/Assign
node name:conv2_3/1/conv2_3/1/bn/beta/read
node name:conv2_3/1/conv2_3/1/bn/moving_mean/Initializer/zeros
node name:conv2_3/1/conv2_3/1/bn/moving_mean
node name:conv2_3/1/conv2_3/1/bn/moving_mean/Assign
node name:conv2_3/1/conv2_3/1/bn/moving_mean/read
node name:conv2_3/1/conv2_3/1/bn/moving_variance/Initializer/ones
node name:conv2_3/1/conv2_3/1/bn/moving_variance
node name:conv2_3/1/conv2_3/1/bn/moving_variance/Assign
node name:conv2_3/1/conv2_3/1/bn/moving_variance/read
node name:conv2_3/1/conv2_3/1/bn/batchnorm/add/y
node name:conv2_3/1/conv2_3/1/bn/batchnorm/add
node name:conv2_3/1/conv2_3/1/bn/batchnorm/Rsqrt
node name:conv2_3/1/conv2_3/1/bn/batchnorm/mul
node name:conv2_3/1/conv2_3/1/bn/batchnorm/mul_1
node name:conv2_3/1/conv2_3/1/bn/batchnorm/sub
node name:conv2_3/1/conv2_3/1/bn/batchnorm/add_1
node name:conv2_3/1/Elu
node name:Dropout_1/Identity
node name:conv2_3/2/weights/Initializer/truncated_normal/shape
node name:conv2_3/2/weights/Initializer/truncated_normal/mean
node name:conv2_3/2/weights/Initializer/truncated_normal/stddev
node name:conv2_3/2/weights/Initializer/truncated_normal/TruncatedNormal
node name:conv2_3/2/weights/Initializer/truncated_normal/mul
node name:conv2_3/2/weights/Initializer/truncated_normal
node name:conv2_3/2/weights
node name:conv2_3/2/weights/Assign
node name:conv2_3/2/weights/read
node name:conv2_3/2/kernel/Regularizer/l2_regularizer/scale
node name:conv2_3/2/kernel/Regularizer/l2_regularizer/L2Loss
node name:conv2_3/2/kernel/Regularizer/l2_regularizer
node name:conv2_3/2/biases/Initializer/zeros
node name:conv2_3/2/biases
node name:conv2_3/2/biases/Assign
node name:conv2_3/2/biases/read
node name:conv2_3/2/convolution/Shape
node name:conv2_3/2/convolution/dilation_rate
node name:conv2_3/2/convolution
node name:conv2_3/2/BiasAdd
node name:add_1
node name:conv3_1/bn/beta/Initializer/zeros
node name:conv3_1/bn/beta
node name:conv3_1/bn/beta/Assign
node name:conv3_1/bn/beta/read
node name:conv3_1/bn/moving_mean/Initializer/zeros
node name:conv3_1/bn/moving_mean
node name:conv3_1/bn/moving_mean/Assign
node name:conv3_1/bn/moving_mean/read
node name:conv3_1/bn/moving_variance/Initializer/ones
node name:conv3_1/bn/moving_variance
node name:conv3_1/bn/moving_variance/Assign
node name:conv3_1/bn/moving_variance/read
node name:conv3_1/bn/batchnorm/add/y
node name:conv3_1/bn/batchnorm/add
node name:conv3_1/bn/batchnorm/Rsqrt
node name:conv3_1/bn/batchnorm/mul
node name:conv3_1/bn/batchnorm/mul_1
node name:conv3_1/bn/batchnorm/sub
node name:conv3_1/bn/batchnorm/add_1
node name:Elu_1
node name:conv3_1/1/weights/Initializer/truncated_normal/shape
node name:conv3_1/1/weights/Initializer/truncated_normal/mean
node name:conv3_1/1/weights/Initializer/truncated_normal/stddev
node name:conv3_1/1/weights/Initializer/truncated_normal/TruncatedNormal
node name:conv3_1/1/weights/Initializer/truncated_normal/mul
node name:conv3_1/1/weights/Initializer/truncated_normal
node name:conv3_1/1/weights
node name:conv3_1/1/weights/Assign
node name:conv3_1/1/weights/read
node name:conv3_1/1/kernel/Regularizer/l2_regularizer/scale
node name:conv3_1/1/kernel/Regularizer/l2_regularizer/L2Loss
node name:conv3_1/1/kernel/Regularizer/l2_regularizer
node name:conv3_1/1/convolution/Shape
node name:conv3_1/1/convolution/dilation_rate
node name:conv3_1/1/convolution
node name:conv3_1/1/conv3_1/1/bn/beta/Initializer/zeros
node name:conv3_1/1/conv3_1/1/bn/beta
node name:conv3_1/1/conv3_1/1/bn/beta/Assign
node name:conv3_1/1/conv3_1/1/bn/beta/read
node name:conv3_1/1/conv3_1/1/bn/moving_mean/Initializer/zeros
node name:conv3_1/1/conv3_1/1/bn/moving_mean
node name:conv3_1/1/conv3_1/1/bn/moving_mean/Assign
node name:conv3_1/1/conv3_1/1/bn/moving_mean/read
node name:conv3_1/1/conv3_1/1/bn/moving_variance/Initializer/ones
node name:conv3_1/1/conv3_1/1/bn/moving_variance
node name:conv3_1/1/conv3_1/1/bn/moving_variance/Assign
node name:conv3_1/1/conv3_1/1/bn/moving_variance/read
node name:conv3_1/1/conv3_1/1/bn/batchnorm/add/y
node name:conv3_1/1/conv3_1/1/bn/batchnorm/add
node name:conv3_1/1/conv3_1/1/bn/batchnorm/Rsqrt
node name:conv3_1/1/conv3_1/1/bn/batchnorm/mul
node name:conv3_1/1/conv3_1/1/bn/batchnorm/mul_1
node name:conv3_1/1/conv3_1/1/bn/batchnorm/sub
node name:conv3_1/1/conv3_1/1/bn/batchnorm/add_1
node name:conv3_1/1/Elu
node name:Dropout_2/Identity
node name:conv3_1/2/weights/Initializer/truncated_normal/shape
node name:conv3_1/2/weights/Initializer/truncated_normal/mean
node name:conv3_1/2/weights/Initializer/truncated_normal/stddev
node name:conv3_1/2/weights/Initializer/truncated_normal/TruncatedNormal
node name:conv3_1/2/weights/Initializer/truncated_normal/mul
node name:conv3_1/2/weights/Initializer/truncated_normal
node name:conv3_1/2/weights
node name:conv3_1/2/weights/Assign
node name:conv3_1/2/weights/read
node name:conv3_1/2/kernel/Regularizer/l2_regularizer/scale
node name:conv3_1/2/kernel/Regularizer/l2_regularizer/L2Loss
node name:conv3_1/2/kernel/Regularizer/l2_regularizer
node name:conv3_1/2/biases/Initializer/zeros
node name:conv3_1/2/biases
node name:conv3_1/2/biases/Assign
node name:conv3_1/2/biases/read
node name:conv3_1/2/convolution/Shape
node name:conv3_1/2/convolution/dilation_rate
node name:conv3_1/2/convolution
node name:conv3_1/2/BiasAdd
node name:conv3_1/projection/weights/Initializer/truncated_normal/shape
node name:conv3_1/projection/weights/Initializer/truncated_normal/mean
node name:conv3_1/projection/weights/Initializer/truncated_normal/stddev
node name:conv3_1/projection/weights/Initializer/truncated_normal/TruncatedNormal
node name:conv3_1/projection/weights/Initializer/truncated_normal/mul
node name:conv3_1/projection/weights/Initializer/truncated_normal
node name:conv3_1/projection/weights
node name:conv3_1/projection/weights/Assign
node name:conv3_1/projection/weights/read
node name:conv3_1/projection/kernel/Regularizer/l2_regularizer/scale
node name:conv3_1/projection/kernel/Regularizer/l2_regularizer/L2Loss
node name:conv3_1/projection/kernel/Regularizer/l2_regularizer
node name:conv3_1/projection/convolution/Shape
node name:conv3_1/projection/convolution/dilation_rate
node name:conv3_1/projection/convolution
node name:add_2
node name:conv3_3/bn/beta/Initializer/zeros
node name:conv3_3/bn/beta
node name:conv3_3/bn/beta/Assign
node name:conv3_3/bn/beta/read
node name:conv3_3/bn/moving_mean/Initializer/zeros
node name:conv3_3/bn/moving_mean
node name:conv3_3/bn/moving_mean/Assign
node name:conv3_3/bn/moving_mean/read
node name:conv3_3/bn/moving_variance/Initializer/ones
node name:conv3_3/bn/moving_variance
node name:conv3_3/bn/moving_variance/Assign
node name:conv3_3/bn/moving_variance/read
node name:conv3_3/bn/batchnorm/add/y
node name:conv3_3/bn/batchnorm/add
node name:conv3_3/bn/batchnorm/Rsqrt
node name:conv3_3/bn/batchnorm/mul
node name:conv3_3/bn/batchnorm/mul_1
node name:conv3_3/bn/batchnorm/sub
node name:conv3_3/bn/batchnorm/add_1
node name:Elu_2
node name:conv3_3/1/weights/Initializer/truncated_normal/shape
node name:conv3_3/1/weights/Initializer/truncated_normal/mean
node name:conv3_3/1/weights/Initializer/truncated_normal/stddev
node name:conv3_3/1/weights/Initializer/truncated_normal/TruncatedNormal
node name:conv3_3/1/weights/Initializer/truncated_normal/mul
node name:conv3_3/1/weights/Initializer/truncated_normal
node name:conv3_3/1/weights
node name:conv3_3/1/weights/Assign
node name:conv3_3/1/weights/read
node name:conv3_3/1/kernel/Regularizer/l2_regularizer/scale
node name:conv3_3/1/kernel/Regularizer/l2_regularizer/L2Loss
node name:conv3_3/1/kernel/Regularizer/l2_regularizer
node name:conv3_3/1/convolution/Shape
node name:conv3_3/1/convolution/dilation_rate
node name:conv3_3/1/convolution
node name:conv3_3/1/conv3_3/1/bn/beta/Initializer/zeros
node name:conv3_3/1/conv3_3/1/bn/beta
node name:conv3_3/1/conv3_3/1/bn/beta/Assign
node name:conv3_3/1/conv3_3/1/bn/beta/read
node name:conv3_3/1/conv3_3/1/bn/moving_mean/Initializer/zeros
node name:conv3_3/1/conv3_3/1/bn/moving_mean
node name:conv3_3/1/conv3_3/1/bn/moving_mean/Assign
node name:conv3_3/1/conv3_3/1/bn/moving_mean/read
node name:conv3_3/1/conv3_3/1/bn/moving_variance/Initializer/ones
node name:conv3_3/1/conv3_3/1/bn/moving_variance
node name:conv3_3/1/conv3_3/1/bn/moving_variance/Assign
node name:conv3_3/1/conv3_3/1/bn/moving_variance/read
node name:conv3_3/1/conv3_3/1/bn/batchnorm/add/y
node name:conv3_3/1/conv3_3/1/bn/batchnorm/add
node name:conv3_3/1/conv3_3/1/bn/batchnorm/Rsqrt
node name:conv3_3/1/conv3_3/1/bn/batchnorm/mul
node name:conv3_3/1/conv3_3/1/bn/batchnorm/mul_1
node name:conv3_3/1/conv3_3/1/bn/batchnorm/sub
node name:conv3_3/1/conv3_3/1/bn/batchnorm/add_1
node name:conv3_3/1/Elu
node name:Dropout_3/Identity
node name:conv3_3/2/weights/Initializer/truncated_normal/shape
node name:conv3_3/2/weights/Initializer/truncated_normal/mean
node name:conv3_3/2/weights/Initializer/truncated_normal/stddev
node name:conv3_3/2/weights/Initializer/truncated_normal/TruncatedNormal
node name:conv3_3/2/weights/Initializer/truncated_normal/mul
node name:conv3_3/2/weights/Initializer/truncated_normal
node name:conv3_3/2/weights
node name:conv3_3/2/weights/Assign
node name:conv3_3/2/weights/read
node name:conv3_3/2/kernel/Regularizer/l2_regularizer/scale
node name:conv3_3/2/kernel/Regularizer/l2_regularizer/L2Loss
node name:conv3_3/2/kernel/Regularizer/l2_regularizer
node name:conv3_3/2/biases/Initializer/zeros
node name:conv3_3/2/biases
node name:conv3_3/2/biases/Assign
node name:conv3_3/2/biases/read
node name:conv3_3/2/convolution/Shape
node name:conv3_3/2/convolution/dilation_rate
node name:conv3_3/2/convolution
node name:conv3_3/2/BiasAdd
node name:add_3
node name:conv4_1/bn/beta/Initializer/zeros
node name:conv4_1/bn/beta
node name:conv4_1/bn/beta/Assign
node name:conv4_1/bn/beta/read
node name:conv4_1/bn/moving_mean/Initializer/zeros
node name:conv4_1/bn/moving_mean
node name:conv4_1/bn/moving_mean/Assign
node name:conv4_1/bn/moving_mean/read
node name:conv4_1/bn/moving_variance/Initializer/ones
node name:conv4_1/bn/moving_variance
node name:conv4_1/bn/moving_variance/Assign
node name:conv4_1/bn/moving_variance/read
node name:conv4_1/bn/batchnorm/add/y
node name:conv4_1/bn/batchnorm/add
node name:conv4_1/bn/batchnorm/Rsqrt
node name:conv4_1/bn/batchnorm/mul
node name:conv4_1/bn/batchnorm/mul_1
node name:conv4_1/bn/batchnorm/sub
node name:conv4_1/bn/batchnorm/add_1
node name:Elu_3
node name:conv4_1/1/weights/Initializer/truncated_normal/shape
node name:conv4_1/1/weights/Initializer/truncated_normal/mean
node name:conv4_1/1/weights/Initializer/truncated_normal/stddev
node name:conv4_1/1/weights/Initializer/truncated_normal/TruncatedNormal
node name:conv4_1/1/weights/Initializer/truncated_normal/mul
node name:conv4_1/1/weights/Initializer/truncated_normal
node name:conv4_1/1/weights
node name:conv4_1/1/weights/Assign
node name:conv4_1/1/weights/read
node name:conv4_1/1/kernel/Regularizer/l2_regularizer/scale
node name:conv4_1/1/kernel/Regularizer/l2_regularizer/L2Loss
node name:conv4_1/1/kernel/Regularizer/l2_regularizer
node name:conv4_1/1/convolution/Shape
node name:conv4_1/1/convolution/dilation_rate
node name:conv4_1/1/convolution
node name:conv4_1/1/conv4_1/1/bn/beta/Initializer/zeros
node name:conv4_1/1/conv4_1/1/bn/beta
node name:conv4_1/1/conv4_1/1/bn/beta/Assign
node name:conv4_1/1/conv4_1/1/bn/beta/read
node name:conv4_1/1/conv4_1/1/bn/moving_mean/Initializer/zeros
node name:conv4_1/1/conv4_1/1/bn/moving_mean
node name:conv4_1/1/conv4_1/1/bn/moving_mean/Assign
node name:conv4_1/1/conv4_1/1/bn/moving_mean/read
node name:conv4_1/1/conv4_1/1/bn/moving_variance/Initializer/ones
node name:conv4_1/1/conv4_1/1/bn/moving_variance
node name:conv4_1/1/conv4_1/1/bn/moving_variance/Assign
node name:conv4_1/1/conv4_1/1/bn/moving_variance/read
node name:conv4_1/1/conv4_1/1/bn/batchnorm/add/y
node name:conv4_1/1/conv4_1/1/bn/batchnorm/add
node name:conv4_1/1/conv4_1/1/bn/batchnorm/Rsqrt
node name:conv4_1/1/conv4_1/1/bn/batchnorm/mul
node name:conv4_1/1/conv4_1/1/bn/batchnorm/mul_1
node name:conv4_1/1/conv4_1/1/bn/batchnorm/sub
node name:conv4_1/1/conv4_1/1/bn/batchnorm/add_1
node name:conv4_1/1/Elu
node name:Dropout_4/Identity
node name:conv4_1/2/weights/Initializer/truncated_normal/shape
node name:conv4_1/2/weights/Initializer/truncated_normal/mean
node name:conv4_1/2/weights/Initializer/truncated_normal/stddev
node name:conv4_1/2/weights/Initializer/truncated_normal/TruncatedNormal
node name:conv4_1/2/weights/Initializer/truncated_normal/mul
node name:conv4_1/2/weights/Initializer/truncated_normal
node name:conv4_1/2/weights
node name:conv4_1/2/weights/Assign
node name:conv4_1/2/weights/read
node name:conv4_1/2/kernel/Regularizer/l2_regularizer/scale
node name:conv4_1/2/kernel/Regularizer/l2_regularizer/L2Loss
node name:conv4_1/2/kernel/Regularizer/l2_regularizer
node name:conv4_1/2/biases/Initializer/zeros
node name:conv4_1/2/biases
node name:conv4_1/2/biases/Assign
node name:conv4_1/2/biases/read
node name:conv4_1/2/convolution/Shape
node name:conv4_1/2/convolution/dilation_rate
node name:conv4_1/2/convolution
node name:conv4_1/2/BiasAdd
node name:conv4_1/projection/weights/Initializer/truncated_normal/shape
node name:conv4_1/projection/weights/Initializer/truncated_normal/mean
node name:conv4_1/projection/weights/Initializer/truncated_normal/stddev
node name:conv4_1/projection/weights/Initializer/truncated_normal/TruncatedNormal
node name:conv4_1/projection/weights/Initializer/truncated_normal/mul
node name:conv4_1/projection/weights/Initializer/truncated_normal
node name:conv4_1/projection/weights
node name:conv4_1/projection/weights/Assign
node name:conv4_1/projection/weights/read
node name:conv4_1/projection/kernel/Regularizer/l2_regularizer/scale
node name:conv4_1/projection/kernel/Regularizer/l2_regularizer/L2Loss
node name:conv4_1/projection/kernel/Regularizer/l2_regularizer
node name:conv4_1/projection/convolution/Shape
node name:conv4_1/projection/convolution/dilation_rate
node name:conv4_1/projection/convolution
node name:add_4
node name:conv4_3/bn/beta/Initializer/zeros
node name:conv4_3/bn/beta
node name:conv4_3/bn/beta/Assign
node name:conv4_3/bn/beta/read
node name:conv4_3/bn/moving_mean/Initializer/zeros
node name:conv4_3/bn/moving_mean
node name:conv4_3/bn/moving_mean/Assign
node name:conv4_3/bn/moving_mean/read
node name:conv4_3/bn/moving_variance/Initializer/ones
node name:conv4_3/bn/moving_variance
node name:conv4_3/bn/moving_variance/Assign
node name:conv4_3/bn/moving_variance/read
node name:conv4_3/bn/batchnorm/add/y
node name:conv4_3/bn/batchnorm/add
node name:conv4_3/bn/batchnorm/Rsqrt
node name:conv4_3/bn/batchnorm/mul
node name:conv4_3/bn/batchnorm/mul_1
node name:conv4_3/bn/batchnorm/sub
node name:conv4_3/bn/batchnorm/add_1
node name:Elu_4
node name:conv4_3/1/weights/Initializer/truncated_normal/shape
node name:conv4_3/1/weights/Initializer/truncated_normal/mean
node name:conv4_3/1/weights/Initializer/truncated_normal/stddev
node name:conv4_3/1/weights/Initializer/truncated_normal/TruncatedNormal
node name:conv4_3/1/weights/Initializer/truncated_normal/mul
node name:conv4_3/1/weights/Initializer/truncated_normal
node name:conv4_3/1/weights
node name:conv4_3/1/weights/Assign
node name:conv4_3/1/weights/read
node name:conv4_3/1/kernel/Regularizer/l2_regularizer/scale
node name:conv4_3/1/kernel/Regularizer/l2_regularizer/L2Loss
node name:conv4_3/1/kernel/Regularizer/l2_regularizer
node name:conv4_3/1/convolution/Shape
node name:conv4_3/1/convolution/dilation_rate
node name:conv4_3/1/convolution
node name:conv4_3/1/conv4_3/1/bn/beta/Initializer/zeros
node name:conv4_3/1/conv4_3/1/bn/beta
node name:conv4_3/1/conv4_3/1/bn/beta/Assign
node name:conv4_3/1/conv4_3/1/bn/beta/read
node name:conv4_3/1/conv4_3/1/bn/moving_mean/Initializer/zeros
node name:conv4_3/1/conv4_3/1/bn/moving_mean
node name:conv4_3/1/conv4_3/1/bn/moving_mean/Assign
node name:conv4_3/1/conv4_3/1/bn/moving_mean/read
node name:conv4_3/1/conv4_3/1/bn/moving_variance/Initializer/ones
node name:conv4_3/1/conv4_3/1/bn/moving_variance
node name:conv4_3/1/conv4_3/1/bn/moving_variance/Assign
node name:conv4_3/1/conv4_3/1/bn/moving_variance/read
node name:conv4_3/1/conv4_3/1/bn/batchnorm/add/y
node name:conv4_3/1/conv4_3/1/bn/batchnorm/add
node name:conv4_3/1/conv4_3/1/bn/batchnorm/Rsqrt
node name:conv4_3/1/conv4_3/1/bn/batchnorm/mul
node name:conv4_3/1/conv4_3/1/bn/batchnorm/mul_1
node name:conv4_3/1/conv4_3/1/bn/batchnorm/sub
node name:conv4_3/1/conv4_3/1/bn/batchnorm/add_1
node name:conv4_3/1/Elu
node name:Dropout_5/Identity
node name:conv4_3/2/weights/Initializer/truncated_normal/shape
node name:conv4_3/2/weights/Initializer/truncated_normal/mean
node name:conv4_3/2/weights/Initializer/truncated_normal/stddev
node name:conv4_3/2/weights/Initializer/truncated_normal/TruncatedNormal
node name:conv4_3/2/weights/Initializer/truncated_normal/mul
node name:conv4_3/2/weights/Initializer/truncated_normal
node name:conv4_3/2/weights
node name:conv4_3/2/weights/Assign
node name:conv4_3/2/weights/read
node name:conv4_3/2/kernel/Regularizer/l2_regularizer/scale
node name:conv4_3/2/kernel/Regularizer/l2_regularizer/L2Loss
node name:conv4_3/2/kernel/Regularizer/l2_regularizer
node name:conv4_3/2/biases/Initializer/zeros
node name:conv4_3/2/biases
node name:conv4_3/2/biases/Assign
node name:conv4_3/2/biases/read
node name:conv4_3/2/convolution/Shape
node name:conv4_3/2/convolution/dilation_rate
node name:conv4_3/2/convolution
node name:conv4_3/2/BiasAdd
node name:add_5
node name:Flatten/Shape
node name:Flatten/Slice/begin
node name:Flatten/Slice/size
node name:Flatten/Slice
node name:Flatten/Slice_1/begin
node name:Flatten/Slice_1/size
node name:Flatten/Slice_1
node name:Flatten/Const
node name:Flatten/Prod
node name:Flatten/ExpandDims/dim
node name:Flatten/ExpandDims
node name:Flatten/concat/axis
node name:Flatten/concat
node name:Flatten/Reshape
node name:Dropout_6/Identity
node name:fc1/weights/Initializer/truncated_normal/shape
node name:fc1/weights/Initializer/truncated_normal/mean
node name:fc1/weights/Initializer/truncated_normal/stddev
node name:fc1/weights/Initializer/truncated_normal/TruncatedNormal
node name:fc1/weights/Initializer/truncated_normal/mul
node name:fc1/weights/Initializer/truncated_normal
node name:fc1/weights
node name:fc1/weights/Assign
node name:fc1/weights/read
node name:fc1/kernel/Regularizer/l2_regularizer/scale
node name:fc1/kernel/Regularizer/l2_regularizer/L2Loss
node name:fc1/kernel/Regularizer/l2_regularizer
node name:fc1/MatMul
node name:fc1/fc1/bn/beta/Initializer/zeros
node name:fc1/fc1/bn/beta
node name:fc1/fc1/bn/beta/Assign
node name:fc1/fc1/bn/beta/read
node name:fc1/fc1/bn/moving_mean/Initializer/zeros
node name:fc1/fc1/bn/moving_mean
node name:fc1/fc1/bn/moving_mean/Assign
node name:fc1/fc1/bn/moving_mean/read
node name:fc1/fc1/bn/moving_variance/Initializer/ones
node name:fc1/fc1/bn/moving_variance
node name:fc1/fc1/bn/moving_variance/Assign
node name:fc1/fc1/bn/moving_variance/read
node name:fc1/fc1/bn/batchnorm/add/y
node name:fc1/fc1/bn/batchnorm/add
node name:fc1/fc1/bn/batchnorm/Rsqrt
node name:fc1/fc1/bn/batchnorm/mul
node name:fc1/fc1/bn/batchnorm/mul_1
node name:fc1/fc1/bn/batchnorm/sub
node name:fc1/fc1/bn/batchnorm/add_1
node name:fc1/Elu
node name:ball/beta/Initializer/zeros
node name:ball/beta
node name:ball/beta/Assign
node name:ball/beta/read
node name:ball/moving_mean/Initializer/zeros
node name:ball/moving_mean
node name:ball/moving_mean/Assign
node name:ball/moving_mean/read
node name:ball/moving_variance/Initializer/ones
node name:ball/moving_variance
node name:ball/moving_variance/Assign
node name:ball/moving_variance/read
node name:ball/batchnorm/add/y
node name:ball/batchnorm/add
node name:ball/batchnorm/Rsqrt
node name:ball/batchnorm/mul
node name:ball/batchnorm/mul_1
node name:ball/batchnorm/sub
node name:ball/batchnorm/add_1
node name:Const
node name:Square
node name:Sum/reduction_indices
node name:Sum
node name:add_6
node name:Sqrt
node name:truediv
node name:ball/mean_vectors/Initializer/truncated_normal/shape
node name:ball/mean_vectors/Initializer/truncated_normal/mean
node name:ball/mean_vectors/Initializer/truncated_normal/stddev
node name:ball/mean_vectors/Initializer/truncated_normal/TruncatedNormal
node name:ball/mean_vectors/Initializer/truncated_normal/mul
node name:ball/mean_vectors/Initializer/truncated_normal
node name:ball/mean_vectors
node name:ball/mean_vectors/Assign
node name:ball/mean_vectors/read
node name:ball/scale/Initializer/Const
node name:ball/scale
node name:ball/scale/Assign
node name:ball/scale/read
node name:ball_1/Softplus
node name:Const_1
node name:Square_1
node name:Sum_1/reduction_indices
node name:Sum_1
node name:add_7
node name:Sqrt_1
node name:truediv_1
node name:MatMul
node name:mul
node name:save/Const
node name:save/SaveV2/tensor_names
node name:save/SaveV2/shape_and_slices
node name:save/SaveV2
node name:save/control_dependency
node name:save/RestoreV2/tensor_names
node name:save/RestoreV2/shape_and_slices
node name:save/RestoreV2
node name:save/Assign
node name:save/RestoreV2_1/tensor_names
node name:save/RestoreV2_1/shape_and_slices
node name:save/RestoreV2_1
node name:save/Assign_1
node name:save/RestoreV2_2/tensor_names
node name:save/RestoreV2_2/shape_and_slices
node name:save/RestoreV2_2
node name:save/Assign_2
node name:save/RestoreV2_3/tensor_names
node name:save/RestoreV2_3/shape_and_slices
node name:save/RestoreV2_3
node name:save/Assign_3
node name:save/RestoreV2_4/tensor_names
node name:save/RestoreV2_4/shape_and_slices
node name:save/RestoreV2_4
node name:save/Assign_4
node name:save/RestoreV2_5/tensor_names
node name:save/RestoreV2_5/shape_and_slices
node name:save/RestoreV2_5
node name:save/Assign_5
node name:save/RestoreV2_6/tensor_names
node name:save/RestoreV2_6/shape_and_slices
node name:save/RestoreV2_6
node name:save/Assign_6
node name:save/RestoreV2_7/tensor_names
node name:save/RestoreV2_7/shape_and_slices
node name:save/RestoreV2_7
node name:save/Assign_7
node name:save/RestoreV2_8/tensor_names
node name:save/RestoreV2_8/shape_and_slices
node name:save/RestoreV2_8
node name:save/Assign_8
node name:save/RestoreV2_9/tensor_names
node name:save/RestoreV2_9/shape_and_slices
node name:save/RestoreV2_9
node name:save/Assign_9
node name:save/RestoreV2_10/tensor_names
node name:save/RestoreV2_10/shape_and_slices
node name:save/RestoreV2_10
node name:save/Assign_10
node name:save/RestoreV2_11/tensor_names
node name:save/RestoreV2_11/shape_and_slices
node name:save/RestoreV2_11
node name:save/Assign_11
node name:save/RestoreV2_12/tensor_names
node name:save/RestoreV2_12/shape_and_slices
node name:save/RestoreV2_12
node name:save/Assign_12
node name:save/RestoreV2_13/tensor_names
node name:save/RestoreV2_13/shape_and_slices
node name:save/RestoreV2_13
node name:save/Assign_13
node name:save/RestoreV2_14/tensor_names
node name:save/RestoreV2_14/shape_and_slices
node name:save/RestoreV2_14
node name:save/Assign_14
node name:save/RestoreV2_15/tensor_names
node name:save/RestoreV2_15/shape_and_slices
node name:save/RestoreV2_15
node name:save/Assign_15
node name:save/RestoreV2_16/tensor_names
node name:save/RestoreV2_16/shape_and_slices
node name:save/RestoreV2_16
node name:save/Assign_16
node name:save/RestoreV2_17/tensor_names
node name:save/RestoreV2_17/shape_and_slices
node name:save/RestoreV2_17
node name:save/Assign_17
node name:save/RestoreV2_18/tensor_names
node name:save/RestoreV2_18/shape_and_slices
node name:save/RestoreV2_18
node name:save/Assign_18
node name:save/RestoreV2_19/tensor_names
node name:save/RestoreV2_19/shape_and_slices
node name:save/RestoreV2_19
node name:save/Assign_19
node name:save/RestoreV2_20/tensor_names
node name:save/RestoreV2_20/shape_and_slices
node name:save/RestoreV2_20
node name:save/Assign_20
node name:save/RestoreV2_21/tensor_names
node name:save/RestoreV2_21/shape_and_slices
node name:save/RestoreV2_21
node name:save/Assign_21
node name:save/RestoreV2_22/tensor_names
node name:save/RestoreV2_22/shape_and_slices
node name:save/RestoreV2_22
node name:save/Assign_22
node name:save/RestoreV2_23/tensor_names
node name:save/RestoreV2_23/shape_and_slices
node name:save/RestoreV2_23
node name:save/Assign_23
node name:save/RestoreV2_24/tensor_names
node name:save/RestoreV2_24/shape_and_slices
node name:save/RestoreV2_24
node name:save/Assign_24
node name:save/RestoreV2_25/tensor_names
node name:save/RestoreV2_25/shape_and_slices
node name:save/RestoreV2_25
node name:save/Assign_25
node name:save/RestoreV2_26/tensor_names
node name:save/RestoreV2_26/shape_and_slices
node name:save/RestoreV2_26
node name:save/Assign_26
node name:save/RestoreV2_27/tensor_names
node name:save/RestoreV2_27/shape_and_slices
node name:save/RestoreV2_27
node name:save/Assign_27
node name:save/RestoreV2_28/tensor_names
node name:save/RestoreV2_28/shape_and_slices
node name:save/RestoreV2_28
node name:save/Assign_28
node name:save/RestoreV2_29/tensor_names
node name:save/RestoreV2_29/shape_and_slices
node name:save/RestoreV2_29
node name:save/Assign_29
node name:save/RestoreV2_30/tensor_names
node name:save/RestoreV2_30/shape_and_slices
node name:save/RestoreV2_30
node name:save/Assign_30
node name:save/RestoreV2_31/tensor_names
node name:save/RestoreV2_31/shape_and_slices
node name:save/RestoreV2_31
node name:save/Assign_31
node name:save/RestoreV2_32/tensor_names
node name:save/RestoreV2_32/shape_and_slices
node name:save/RestoreV2_32
node name:save/Assign_32
node name:save/RestoreV2_33/tensor_names
node name:save/RestoreV2_33/shape_and_slices
node name:save/RestoreV2_33
node name:save/Assign_33
node name:save/RestoreV2_34/tensor_names
node name:save/RestoreV2_34/shape_and_slices
node name:save/RestoreV2_34
node name:save/Assign_34
node name:save/RestoreV2_35/tensor_names
node name:save/RestoreV2_35/shape_and_slices
node name:save/RestoreV2_35
node name:save/Assign_35
node name:save/RestoreV2_36/tensor_names
node name:save/RestoreV2_36/shape_and_slices
node name:save/RestoreV2_36
node name:save/Assign_36
node name:save/RestoreV2_37/tensor_names
node name:save/RestoreV2_37/shape_and_slices
node name:save/RestoreV2_37
node name:save/Assign_37
node name:save/RestoreV2_38/tensor_names
node name:save/RestoreV2_38/shape_and_slices
node name:save/RestoreV2_38
node name:save/Assign_38
node name:save/RestoreV2_39/tensor_names
node name:save/RestoreV2_39/shape_and_slices
node name:save/RestoreV2_39
node name:save/Assign_39
node name:save/RestoreV2_40/tensor_names
node name:save/RestoreV2_40/shape_and_slices
node name:save/RestoreV2_40
node name:save/Assign_40
node name:save/RestoreV2_41/tensor_names
node name:save/RestoreV2_41/shape_and_slices
node name:save/RestoreV2_41
node name:save/Assign_41
node name:save/RestoreV2_42/tensor_names
node name:save/RestoreV2_42/shape_and_slices
node name:save/RestoreV2_42
node name:save/Assign_42
node name:save/RestoreV2_43/tensor_names
node name:save/RestoreV2_43/shape_and_slices
node name:save/RestoreV2_43
node name:save/Assign_43
node name:save/RestoreV2_44/tensor_names
node name:save/RestoreV2_44/shape_and_slices
node name:save/RestoreV2_44
node name:save/Assign_44
node name:save/RestoreV2_45/tensor_names
node name:save/RestoreV2_45/shape_and_slices
node name:save/RestoreV2_45
node name:save/Assign_45
node name:save/RestoreV2_46/tensor_names
node name:save/RestoreV2_46/shape_and_slices
node name:save/RestoreV2_46
node name:save/Assign_46
node name:save/RestoreV2_47/tensor_names
node name:save/RestoreV2_47/shape_and_slices
node name:save/RestoreV2_47
node name:save/Assign_47
node name:save/RestoreV2_48/tensor_names
node name:save/RestoreV2_48/shape_and_slices
node name:save/RestoreV2_48
node name:save/Assign_48
node name:save/RestoreV2_49/tensor_names
node name:save/RestoreV2_49/shape_and_slices
node name:save/RestoreV2_49
node name:save/Assign_49
node name:save/RestoreV2_50/tensor_names
node name:save/RestoreV2_50/shape_and_slices
node name:save/RestoreV2_50
node name:save/Assign_50
node name:save/RestoreV2_51/tensor_names
node name:save/RestoreV2_51/shape_and_slices
node name:save/RestoreV2_51
node name:save/Assign_51
node name:save/RestoreV2_52/tensor_names
node name:save/RestoreV2_52/shape_and_slices
node name:save/RestoreV2_52
node name:save/Assign_52
node name:save/RestoreV2_53/tensor_names
node name:save/RestoreV2_53/shape_and_slices
node name:save/RestoreV2_53
node name:save/Assign_53
node name:save/RestoreV2_54/tensor_names
node name:save/RestoreV2_54/shape_and_slices
node name:save/RestoreV2_54
node name:save/Assign_54
node name:save/RestoreV2_55/tensor_names
node name:save/RestoreV2_55/shape_and_slices
node name:save/RestoreV2_55
node name:save/Assign_55
node name:save/RestoreV2_56/tensor_names
node name:save/RestoreV2_56/shape_and_slices
node name:save/RestoreV2_56
node name:save/Assign_56
node name:save/RestoreV2_57/tensor_names
node name:save/RestoreV2_57/shape_and_slices
node name:save/RestoreV2_57
node name:save/Assign_57
node name:save/RestoreV2_58/tensor_names
node name:save/RestoreV2_58/shape_and_slices
node name:save/RestoreV2_58
node name:save/Assign_58
node name:save/RestoreV2_59/tensor_names
node name:save/RestoreV2_59/shape_and_slices
node name:save/RestoreV2_59
node name:save/Assign_59
node name:save/RestoreV2_60/tensor_names
node name:save/RestoreV2_60/shape_and_slices
node name:save/RestoreV2_60
node name:save/Assign_60
node name:save/RestoreV2_61/tensor_names
node name:save/RestoreV2_61/shape_and_slices
node name:save/RestoreV2_61
node name:save/Assign_61
node name:save/RestoreV2_62/tensor_names
node name:save/RestoreV2_62/shape_and_slices
node name:save/RestoreV2_62
node name:save/Assign_62
node name:save/RestoreV2_63/tensor_names
node name:save/RestoreV2_63/shape_and_slices
node name:save/RestoreV2_63
node name:save/Assign_63
node name:save/RestoreV2_64/tensor_names
node name:save/RestoreV2_64/shape_and_slices
node name:save/RestoreV2_64
node name:save/Assign_64
node name:save/RestoreV2_65/tensor_names
node name:save/RestoreV2_65/shape_and_slices
node name:save/RestoreV2_65
node name:save/Assign_65
node name:save/RestoreV2_66/tensor_names
node name:save/RestoreV2_66/shape_and_slices
node name:save/RestoreV2_66
node name:save/Assign_66
node name:save/RestoreV2_67/tensor_names
node name:save/RestoreV2_67/shape_and_slices
node name:save/RestoreV2_67
node name:save/Assign_67
node name:save/RestoreV2_68/tensor_names
node name:save/RestoreV2_68/shape_and_slices
node name:save/RestoreV2_68
node name:save/Assign_68
node name:save/RestoreV2_69/tensor_names
node name:save/RestoreV2_69/shape_and_slices
node name:save/RestoreV2_69
node name:save/Assign_69
node name:save/restore_all
init succeed
Tensorflow get feature succeed!
Detections size:5
Tensorflow get feature succeed!
Detections size:5
Tensorflow get feature succeed!
Detections size:5
Tensorflow get feature succeed!
Detections size:6
Tensorflow get feature succeed!
Detections size:5
Tensorflow get feature succeed!
Detections size:5
Tensorflow get feature succeed!
Detections size:5
Tensorflow get feature succeed!
Detections size:5
Tensorflow get feature succeed!
Detections size:6
Tensorflow get feature succeed!
Detections size:6
Tensorflow get feature succeed!
Detections size:6
Tensorflow get feature succeed!
Detections size:6
Tensorflow get feature succeed!
Detections size:7
Tensorflow get feature succeed!
Detections size:6
Tensorflow get feature succeed!
Detections size:6
Tensorflow get feature succeed!
Detections size:7
Tensorflow get feature succeed!
Detections size:7
Tensorflow get feature succeed!
Detections size:7
Tensorflow get feature succeed!
Detections size:6
Tensorflow get feature succeed!
Detections size:6
Tensorflow get feature succeed!
Detections size:6
Tensorflow get feature succeed!
Detections size:6
Tensorflow get feature succeed!
Detections size:7
Tensorflow get feature succeed!
Detections size:7
Tensorflow get feature succeed!
Detections size:7
Tensorflow get feature succeed!
Detections size:6
Tensorflow get feature succeed!
Detections size:6
Tensorflow get feature succeed!
Detections size:5
Tensorflow get feature succeed!
Detections size:5
Tensorflow get feature succeed!
Detections size:6
Tensorflow get feature succeed!
Detections size:5
Tensorflow get feature succeed!
Detections size:5
Tensorflow get feature succeed!
Detections size:6
Tensorflow get feature succeed!
Detections size:5
Tensorflow get feature succeed!
Detections size:6
Tensorflow get feature succeed!
Detections size:7
Tensorflow get feature succeed!
Detections size:7
Tensorflow get feature succeed!
Detections size:6
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:7
Tensorflow get feature succeed!
Detections size:7
Tensorflow get feature succeed!
Detections size:7
Tensorflow get feature succeed!
Detections size:7
Tensorflow get feature succeed!
Detections size:7
Tensorflow get feature succeed!
Detections size:7
Tensorflow get feature succeed!
Detections size:7
Tensorflow get feature succeed!
Detections size:7
Tensorflow get feature succeed!
Detections size:7
Tensorflow get feature succeed!
Detections size:6
Tensorflow get feature succeed!
Detections size:6
Tensorflow get feature succeed!
Detections size:7
Tensorflow get feature succeed!
Detections size:6
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:6
Tensorflow get feature succeed!
Detections size:7
Tensorflow get feature succeed!
Detections size:6
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:7
Tensorflow get feature succeed!
Detections size:7
Tensorflow get feature succeed!
Detections size:7
Tensorflow get feature succeed!
Detections size:7
Tensorflow get feature succeed!
Detections size:7
Tensorflow get feature succeed!
Detections size:7
Tensorflow get feature succeed!
Detections size:7
Tensorflow get feature succeed!
Detections size:7
Tensorflow get feature succeed!
Detections size:7
Tensorflow get feature succeed!
Detections size:7
Tensorflow get feature succeed!
Detections size:7
Tensorflow get feature succeed!
Detections size:7
Tensorflow get feature succeed!
Detections size:7
Tensorflow get feature succeed!
Detections size:7
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:7
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:7
Tensorflow get feature succeed!
Detections size:7
Tensorflow get feature succeed!
Detections size:7
Tensorflow get feature succeed!
Detections size:7
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:7
Tensorflow get feature succeed!
Detections size:7
Tensorflow get feature succeed!
Detections size:7
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:10
Tensorflow get feature succeed!
Detections size:10
Tensorflow get feature succeed!
Detections size:10
Tensorflow get feature succeed!
Detections size:11
Tensorflow get feature succeed!
Detections size:11
Tensorflow get feature succeed!
Detections size:10
Tensorflow get feature succeed!
Detections size:10
Tensorflow get feature succeed!
Detections size:11
Tensorflow get feature succeed!
Detections size:10
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:10
Tensorflow get feature succeed!
Detections size:10
Tensorflow get feature succeed!
Detections size:10
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:10
Tensorflow get feature succeed!
Detections size:10
Tensorflow get feature succeed!
Detections size:10
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:10
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:10
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:10
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:10
Tensorflow get feature succeed!
Detections size:10
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:10
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:10
Tensorflow get feature succeed!
Detections size:11
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:10
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:7
Tensorflow get feature succeed!
Detections size:7
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:10
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:7
Tensorflow get feature succeed!
Detections size:7
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:7
Tensorflow get feature succeed!
Detections size:7
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:7
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:7
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:7
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:7
Tensorflow get feature succeed!
Detections size:7
Tensorflow get feature succeed!
Detections size:7
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:7
Tensorflow get feature succeed!
Detections size:7
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:7
Tensorflow get feature succeed!
Detections size:7
Tensorflow get feature succeed!
Detections size:7
Tensorflow get feature succeed!
Detections size:7
Tensorflow get feature succeed!
Detections size:6
Tensorflow get feature succeed!
Detections size:7
Tensorflow get feature succeed!
Detections size:7
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:7
Tensorflow get feature succeed!
Detections size:7
Tensorflow get feature succeed!
Detections size:7
Tensorflow get feature succeed!
Detections size:7
Tensorflow get feature succeed!
Detections size:7
Tensorflow get feature succeed!
Detections size:7
Tensorflow get feature succeed!
Detections size:7
Tensorflow get feature succeed!
Detections size:7
Tensorflow get feature succeed!
Detections size:6
Tensorflow get feature succeed!
Detections size:7
Tensorflow get feature succeed!
Detections size:7
Tensorflow get feature succeed!
Detections size:6
Tensorflow get feature succeed!
Detections size:6
Tensorflow get feature succeed!
Detections size:6
Tensorflow get feature succeed!
Detections size:6
Tensorflow get feature succeed!
Detections size:6
Tensorflow get feature succeed!
Detections size:6
Tensorflow get feature succeed!
Detections size:6
Tensorflow get feature succeed!
Detections size:6
Tensorflow get feature succeed!
Detections size:6
Tensorflow get feature succeed!
Detections size:6
Tensorflow get feature succeed!
Detections size:6
Tensorflow get feature succeed!
Detections size:6
Tensorflow get feature succeed!
Detections size:6
Tensorflow get feature succeed!
Detections size:6
Tensorflow get feature succeed!
Detections size:6
Tensorflow get feature succeed!
Detections size:6
Tensorflow get feature succeed!
Detections size:6
Tensorflow get feature succeed!
Detections size:6
Tensorflow get feature succeed!
Detections size:6
Tensorflow get feature succeed!
Detections size:5
Tensorflow get feature succeed!
Detections size:5
Tensorflow get feature succeed!
Detections size:5
Tensorflow get feature succeed!
Detections size:5
Tensorflow get feature succeed!
Detections size:5
Tensorflow get feature succeed!
Detections size:5
Tensorflow get feature succeed!
Detections size:5
Tensorflow get feature succeed!
Detections size:5
Tensorflow get feature succeed!
Detections size:5
Tensorflow get feature succeed!
Detections size:4
Tensorflow get feature succeed!
Detections size:4
Tensorflow get feature succeed!
Detections size:4
Tensorflow get feature succeed!
Detections size:4
Tensorflow get feature succeed!
Detections size:4
Tensorflow get feature succeed!
Detections size:4
Tensorflow get feature succeed!
Detections size:4
Tensorflow get feature succeed!
Detections size:4
Tensorflow get feature succeed!
Detections size:4
Tensorflow get feature succeed!
Detections size:4
Tensorflow get feature succeed!
Detections size:4
Tensorflow get feature succeed!
Detections size:4
Tensorflow get feature succeed!
Detections size:4
Tensorflow get feature succeed!
Detections size:5
Tensorflow get feature succeed!
Detections size:5
Tensorflow get feature succeed!
Detections size:5
Tensorflow get feature succeed!
Detections size:5
Tensorflow get feature succeed!
Detections size:5
Tensorflow get feature succeed!
Detections size:5
Tensorflow get feature succeed!
Detections size:5
Tensorflow get feature succeed!
Detections size:5
Tensorflow get feature succeed!
Detections size:5
Tensorflow get feature succeed!
Detections size:5
Tensorflow get feature succeed!
Detections size:5
Tensorflow get feature succeed!
Detections size:5
Tensorflow get feature succeed!
Detections size:5
Tensorflow get feature succeed!
Detections size:5
Tensorflow get feature succeed!
Detections size:5
Tensorflow get feature succeed!
Detections size:5
Tensorflow get feature succeed!
Detections size:5
Tensorflow get feature succeed!
Detections size:5
Tensorflow get feature succeed!
Detections size:6
Tensorflow get feature succeed!
Detections size:5
Tensorflow get feature succeed!
Detections size:5
Tensorflow get feature succeed!
Detections size:5
Tensorflow get feature succeed!
Detections size:5
Tensorflow get feature succeed!
Detections size:5
Tensorflow get feature succeed!
Detections size:5
Tensorflow get feature succeed!
Detections size:5
Tensorflow get feature succeed!
Detections size:5
Tensorflow get feature succeed!
Detections size:5
Tensorflow get feature succeed!
Detections size:5
Tensorflow get feature succeed!
Detections size:5
Tensorflow get feature succeed!
Detections size:5
Tensorflow get feature succeed!
Detections size:5
Tensorflow get feature succeed!
Detections size:5
Tensorflow get feature succeed!
Detections size:5
Tensorflow get feature succeed!
Detections size:5
Tensorflow get feature succeed!
Detections size:5
Tensorflow get feature succeed!
Detections size:5
Tensorflow get feature succeed!
Detections size:5
Tensorflow get feature succeed!
Detections size:5
Tensorflow get feature succeed!
Detections size:5
Tensorflow get feature succeed!
Detections size:5
Tensorflow get feature succeed!
Detections size:5
Tensorflow get feature succeed!
Detections size:5
Tensorflow get feature succeed!
Detections size:5
Tensorflow get feature succeed!
Detections size:5
Tensorflow get feature succeed!
Detections size:6
Tensorflow get feature succeed!
Detections size:6
Tensorflow get feature succeed!
Detections size:6
Tensorflow get feature succeed!
Detections size:6
Tensorflow get feature succeed!
Detections size:6
Tensorflow get feature succeed!
Detections size:6
Tensorflow get feature succeed!
Detections size:6
Tensorflow get feature succeed!
Detections size:6
Tensorflow get feature succeed!
Detections size:6
Tensorflow get feature succeed!
Detections size:6
Tensorflow get feature succeed!
Detections size:6
Tensorflow get feature succeed!
Detections size:5
Tensorflow get feature succeed!
Detections size:5
Tensorflow get feature succeed!
Detections size:5
Tensorflow get feature succeed!
Detections size:5
Tensorflow get feature succeed!
Detections size:5
Tensorflow get feature succeed!
Detections size:5
Tensorflow get feature succeed!
Detections size:5
Tensorflow get feature succeed!
Detections size:6
Tensorflow get feature succeed!
Detections size:6
Tensorflow get feature succeed!
Detections size:6
Tensorflow get feature succeed!
Detections size:5
Tensorflow get feature succeed!
Detections size:5
Tensorflow get feature succeed!
Detections size:5
Tensorflow get feature succeed!
Detections size:6
Tensorflow get feature succeed!
Detections size:6
Tensorflow get feature succeed!
Detections size:6
Tensorflow get feature succeed!
Detections size:6
Tensorflow get feature succeed!
Detections size:6
Tensorflow get feature succeed!
Detections size:6
Tensorflow get feature succeed!
Detections size:6
Tensorflow get feature succeed!
Detections size:6
Tensorflow get feature succeed!
Detections size:6
Tensorflow get feature succeed!
Detections size:6
Tensorflow get feature succeed!
Detections size:5
Tensorflow get feature succeed!
Detections size:6
Tensorflow get feature succeed!
Detections size:5
Tensorflow get feature succeed!
Detections size:6
Tensorflow get feature succeed!
Detections size:6
Tensorflow get feature succeed!
Detections size:6
Tensorflow get feature succeed!
Detections size:7
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:7
Tensorflow get feature succeed!
Detections size:7
Tensorflow get feature succeed!
Detections size:7
Tensorflow get feature succeed!
Detections size:7
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:7
Tensorflow get feature succeed!
Detections size:7
Tensorflow get feature succeed!
Detections size:7
Tensorflow get feature succeed!
Detections size:7
Tensorflow get feature succeed!
Detections size:7
Tensorflow get feature succeed!
Detections size:7
Tensorflow get feature succeed!
Detections size:7
Tensorflow get feature succeed!
Detections size:7
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:7
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:7
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:7
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:7
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:7
Tensorflow get feature succeed!
Detections size:7
Tensorflow get feature succeed!
Detections size:7
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:10
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:7
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:6
Tensorflow get feature succeed!
Detections size:6
Tensorflow get feature succeed!
Detections size:7
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:7
Tensorflow get feature succeed!
Detections size:10
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:10
Tensorflow get feature succeed!
Detections size:10
Tensorflow get feature succeed!
Detections size:10
Tensorflow get feature succeed!
Detections size:11
Tensorflow get feature succeed!
Detections size:10
Tensorflow get feature succeed!
Detections size:10
Tensorflow get feature succeed!
Detections size:10
Tensorflow get feature succeed!
Detections size:10
Tensorflow get feature succeed!
Detections size:10
Tensorflow get feature succeed!
Detections size:10
Tensorflow get feature succeed!
Detections size:10
Tensorflow get feature succeed!
Detections size:10
Tensorflow get feature succeed!
Detections size:10
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:10
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:10
Tensorflow get feature succeed!
Detections size:10
Tensorflow get feature succeed!
Detections size:10
Tensorflow get feature succeed!
Detections size:10
Tensorflow get feature succeed!
Detections size:10
Tensorflow get feature succeed!
Detections size:10
Tensorflow get feature succeed!
Detections size:10
Tensorflow get feature succeed!
Detections size:10
Tensorflow get feature succeed!
Detections size:10
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:10
Tensorflow get feature succeed!
Detections size:10
Tensorflow get feature succeed!
Detections size:10
Tensorflow get feature succeed!
Detections size:10
Tensorflow get feature succeed!
Detections size:10
Tensorflow get feature succeed!
Detections size:10
Tensorflow get feature succeed!
Detections size:10
Tensorflow get feature succeed!
Detections size:10
Tensorflow get feature succeed!
Detections size:10
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:10
Tensorflow get feature succeed!
Detections size:10
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:10
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:7
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:7
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:8
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Detections size:9
Tensorflow get feature succeed!
Done processing !!!
Output file is stored as run_yolo_out_cpp.avi
# Deepsort
# Deepsort
