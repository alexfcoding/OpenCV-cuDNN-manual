# Compiling OpenCV with CUDA GPU acceleration in Ubuntu 20.04 LTS and Python virtual environment for neural network development
### Mask R-CNN example video
[![Mask R-CNN example](previewYtb.jpg?raw=true)](https://youtu.be/JomFBZoPjaM "Mask R-CNN")

*OS: Ubuntu 20.04 LTS*

*GPU: NVIDIA RTX 2060*

*NVIDIA driver: 440*

*OpenCV: 4.3*

*CUDA: 10.0*

*cuDNN: 7.6.4*

*Python: 3.8*

### Update system:	
	$ sudo apt-get update
	$ sudo apt-get upgrade
### Install NVIDIA driver:
	$ ubuntu-drivers devices
	$ sudo ubuntu-drivers autoinstall
	$ sudo reboot
### or:
	$ sudo apt install nvidia-440
	$ sudo reboot
### Check GPU:
	nvidia-smi
### Install libraries:
	$ sudo apt-get install libxmu-dev libxi-dev libglu1-mesa libglu1-mesa-dev
	$ sudo apt-get install libjpeg-dev libpng-dev libtiff-dev
	$ sudo apt-get install libavcodec-dev libavformat-dev libswscale-dev libv4l-dev
	$ sudo apt-get install libxvidcore-dev libx264-dev
	$ sudo apt-get install libopenblas-dev libatlas-base-dev liblapack-dev gfortran
	$ sudo apt-get install libhdf5-serial-dev
### Python 3:
	$ sudo apt-get install python3-dev python3-tk python-imaging-tk
	$ sudo apt-get install libgtk-3-dev
### Download and install CUDA 10.0:
	$ cd ~
	$ mkdir tempDir
	$ cd tempDir/
	$ wget https://developer.nvidia.com/compute/cuda/10.0/Prod/local_installers/cuda_10.0.130_410.48_linux
	$ mv cuda_10.0.130_410.48_linux cuda_10.0.130_410.48_linux.run
	$ chmod +x cuda_10.0.130_410.48_linux.run
	$ sudo ./cuda_10.0.130_410.48_linux.run --override
### Bash setup:
	$ nano ~/.bashrc
### Insert this at the bottom of profile:
	# NVIDIA CUDA Toolkit
	export PATH=/usr/local/cuda-10.0/bin:$PATH
	export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64
### Source profile:
	$ source ~/.bashrc
### Check CUDA install:
	$ nvcc -V
### Download cuDNN v7.6.4, for CUDA 10.0:
	https://developer.nvidia.com/rdp/cudnn-archive
	"cuDNN Library for Linux"
### Install cuDNN:
	$ cd ~/tempDir
	$ tar -zxf cudnn-10.0-linux-x64-v7.6.4.38.tgz
	$ cd cuda
	$ sudo cp -P lib64/* /usr/local/cuda/lib64/
	$ sudo cp -P include/* /usr/local/cuda/include/
	$ cd ~
### Download and unpack OpenCV 4.3:	
	$ cd ~
	$ wget -O opencv.zip https://github.com/opencv/opencv/archive/4.3.0.zip
	$ wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/4.3.0.zip
	$ unzip opencv.zip
	$ unzip opencv_contrib.zip
	$ mv opencv-4.2.0 opencv
	$ mv opencv_contrib-4.2.0 opencv_contrib
### Setup Python virtual environment:
	$ wget https://bootstrap.pypa.io/get-pip.py
	$ sudo python3 get-pip.py
	$ sudo pip install virtualenv virtualenvwrapper
	$ sudo rm -rf ~/get-pip.py ~/.cache/pip
	$ nano ~/.bashrc
### Insert this at the bottom of profile:
	# virtualenv and virtualenvwrapper
	export WORKON_HOME=$HOME/.virtualenvs
	export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python3
	source /usr/local/bin/virtualenvwrapper.sh
### reload file:
	$ source ~/.bashrc
### Create Python environment:
	$ mkvirtualenv opencv_gpu -p python33
	$ pip install numpy
	$ workon opencv_gpu
### Find architecture version for your GPU:
	https://developer.nvidia.com/cuda-gpus
### Configure OpenCV with CUDA+cuDNN:
	$ cd ~/opencv
	$ mkdir build
	$ cd build
### type "CUDA_ARCH_BIN" with architecture version of your GPU (for RTX 2060 it is "7.5"):
	$ cmake -D CMAKE_BUILD_TYPE=RELEASE \
	-D CMAKE_INSTALL_PREFIX=/usr/local \
	-D INSTALL_PYTHON_EXAMPLES=ON \
	-D INSTALL_C_EXAMPLES=OFF \
	-D OPENCV_ENABLE_NONFREE=ON \
	-D WITH_CUDA=ON \
	-D WITH_CUDNN=ON \
	-D OPENCV_DNN_CUDA=ON \
	-D ENABLE_FAST_MATH=1 \
	-D CUDA_FAST_MATH=1 \
	-D CUDA_ARCH_BIN=7.5 \
	-D WITH_CUBLAS=1 \
	-D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib/modules \
	-D HAVE_opencv_python3=ON \
	-D PYTHON_EXECUTABLE=~/.virtualenvs/opencv_gpu/bin/python \
	-D BUILD_EXAMPLES=ON ..
### Modules included:
	OpenCV modules:
	-- To be built:	aruco bgsegm bioinspired calib3d ccalib core cudaarithm cudabgsegm cudacodec cudafeatures2d cudafilters cudaimgproc cudalegacy cudaobjdetect cudaoptflow cudastereo cudawarping cudev datasets dnn dnn_objdetect dnn_superres dpm face features2d flann freetype fuzzy 	gapi hdf hfs highgui img_hash imgcodecs imgproc intensity_transform line_descriptor ml objdetect optflow phase_unwrapping photo plot python3 quality rapid reg rgbd saliency shape stereo stitching structured_light superres surface_matching text tracking ts video videoio videostab xfeatures2d ximgproc 	xobjdetect xphoto
### Remember install path:
	--   Python 3:
	--     Interpreter:                 /home/al/.virtualenvs/opencv_gpu/bin/python3 (ver 3.8.2)
	--     Libraries:                   /usr/lib/x86_64-linux-gnu/libpython3.8.so (ver 3.8.2)
	--     numpy:                       /home/al/.virtualenvs/opencv_gpu/lib/python3.8/site-packages/numpy/core/include (ver 1.18.4)
	--     install path:                lib/python3.8/site-packages/cv2/python-3.8
### Compile OpenCV with CUDA+cuDNN (make -jx, where x - number of CPU threads):
	$ make -j12
### On Ubuntu 20.04 LTS it will cause error:
	unsupported GNU version! gcc versions later than 7 are not supported!
### So we need to downgrade gcc version or select alternative:
	$ sudo apt install build-essential
	$ sudo apt -y install gcc-7 g++-7 gcc-8 g++-8 gcc-9 g++-9
	$ sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 7
	$ sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-7 7
	$ sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-8 8
	$ sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-8 8
	$ sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 9
	$ sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-9 9
### Select gcc version (we need gcc-7) by typing number "1":
	sudo update-alternatives --config gcc
### Check current gcc version:
	$ gcc --version
	gcc (Ubuntu 7.5.0-6ubuntu2) 7.5.0
  	Copyright (C) 2017 Free Software Foundation, Inc.
  	This is free software; see the source for copying conditions.  There is NO
  	warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
### If it is v7, repeat command:
	$ make -j12
### Now we can install OpenCV:
	$ sudo make install
	$ sudo ldconfig
### Use saved python install path (lib/python3.8/site-packages/cv2/python-3.8) to check OpenCV bindings file name:
	ls -l /usr/local/lib/python3.8/site-packages/cv2/python-3.8
	"cv2.cpython-38-x86_64-linux-gnu.so"
### Create symbolic link from OpenCV to Python virtual environment:
	$ cd ~/.virtualenvs/opencv_gpu/lib/python3.8/site-packages/
	$ ln -s /usr/local/lib/python3.8/site-packages/cv2/python-3.8/cv2.cpython-38-x86_64-linux-gnu.so cv2.so
### Verify OpenCV with cuDNN in Python:
	$ workon opencv_gpu
	$ python
	Python 3.8.2 (default, Apr 27 2020, 15:53:34) 
	[GCC 9.3.0] on linux
	Type "help", "copyright", "credits" or "license" for more information.
	>>> import cv2
	>>> cv2.__version__
	'4.3.0'
### Use this commands to enable GPU acceleration for your network:
	myNetwork.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
	myNetwork.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
