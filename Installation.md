# I. Installation Steps for Intel Realsense D435i camera module

**Note:** This installation guide it was tested for **Raspberry PI4 ModelB 4GB RAM** and **NVIDIA Jetson Nano B01 4GB RAM**. For full installation steps visit: https://github.com/IntelRealSense/librealsense/blob/master/doc/installation.md
## OS Versions: 
* On Raspberry PI: Raspbian 64bit (beta).
* On Jetson Nano: Image provided by Nvidia(Ubuntu 18.04).
## 1.Make Ubuntu Up-to-date:
**Note:** **Only on RPI**, because Jetson Nano has some problems with the updating package so it's not recommended(display will be gone).
  * Update Ubuntu distribution, including getting the latest stable kernel:
    * `sudo apt-get update && sudo apt-get upgrade && sudo apt-get dist-upgrade`  <br />  

## 2. Download librealsense github repository: 
* Download the complete source tree with *git*<br />
  `git clone https://github.com/IntelRealSense/librealsense.git`<br />

## 3. Prepare Linux Backend and the Dev. Environment: 
  1. Navigate to *librealsense* root directory to run the following scripts.<br />
     ***Unplug any connected Intel RealSense camera.***<br />  

  2. Install the core packages required to build *librealsense* binaries and the affected kernel modules:  
    `sudo apt-get install git libssl-dev libusb-1.0-0-dev pkg-config libgtk-3-dev`  <br /><br />

     * Then for both:<br />
      `sudo apt-get install libglfw3-dev libgl1-mesa-dev libglu1-mesa-dev at`  <br /><br />
    
> **Cmake Note**: certain librealsense CMAKE flags (e.g. CUDA) require version 3.8+ which is currently not made available via apt manager for Ubuntu LTS.   
    Go to the [official CMake site](https://cmake.org/download/) to download and install the application  

  **Note** on graphic sub-system utilization:<br />
     *glfw3*, *mesa* and *gtk* packages are required if you plan to build the SDK's OpenGL-enabled examples. The *librealsense* core library and a range of demos/tools are designed for headless environment deployment.

## 4. Run Intel Realsense permissions script:
* Make sure you're in the librealsense root directory then run:  
    `./scripts/setup_udev_rules.sh`  
<br />
*Notice: One can always remove permissions by running:* *`./scripts/setup_udev_rules.sh --uninstall`*

## 5. Building librealsense2 SDK
  * Navigate to *librealsense* root directory and run `mkdir build && cd build`<br />
  * Run CMake:
    * `cmake ../` - The default build is set to produce the core shared object and unit-tests binaries in Debug mode. Use `-DCMAKE_BUILD_TYPE=Release` to build with optimizations.<br />
    * `cmake ../ -DCMAKE_BUILD_TYPE=release -DBUILD_EXAMPLES=true -DBUILD_GRAPHICAL_EXAMPLES=true` - Builds *librealsense* along with the demos and tutorials<br />

  * Recompile and install *librealsense* binaries:<br />  
  `sudo make uninstall && make clean && make -j4 && sudo make install`<br />  
  The shared object will be installed in `/usr/local/lib`, header files in `/usr/local/include`.<br />
  The binary demos, tutorials and test files will be copied into `/usr/local/bin`<br />
  **Tip:** Use *`make -jX`* for parallel compilation, where *`X`* stands for the number of CPU cores available:<br />
  This enhancement may significantly improve the build time. The side-effect, however, is that it may cause a low-end platform to hang randomly.<br />

## 6. Turn on the camera
* Connect your IntelRealsense D435i to the USB port and then open a terminal and type:
`realsense-viewer` <br/>
This will start the GUI application of IntelRealsense D435i and the rest will be history.

 
# II. Install LXDE as default desktop to save ~1GB of RAM.
## 1. Remove ubuntu-desktop
* `sudo apt remove --purge ubuntu-desktop`
## 2. Install lxdm display manager. It may prompt a dialog to choose a display manager, choose lxdm.
* `sudo apt install lxdm`

## 3. Remove Ubuntu Unity's default display manager gdm3.
* `sudo apt remove --purge gdm3`

## 4. Install LXDE desktop environment.
* `sudo apt install lxde`

## 5. It is recommended to reinstall lxdm to reconfigure for lubuntu-desktop
* `sudo apt install --reinstall lxdm`

## 6. Reboot the system in order to apply the changes.
* `sudo reboot`
   
 <br />  

# III. Enable Remote Desktop Connection from other device.
## 1. Make sure ssh is enabled.
* `sudo apt install openssh-server`
* `sudo systemctl status ssh`

It should be green and running.

## 2. Ubuntu comes with a firewall configuration tool called UFW. If the firewall is enabled on your system, make sure to open the SSH port:
* `sudo ufw allow ssh`

## 3. Test the connection. From another device run:
* `ssh <user>@<ip_address>`
In case of WARNING: REMOTE HOST IDENTIFICATION HAS CHANGED! use ```ssh-keygen -R "you server hostname or ip"```

## 4. Install XRDP.
* `sudo apt-get install xrdp`

## 5. Based on steps from II stage, you must have LXDE installed by now. So now we must adjust the xrdp configuration
Open up /etc/xrdp/startwm.sh:
* `sudo nano /etc/xrdp/startwm.sh`

**IMPORTANT** Comment out the last two lines(for NVIDIA Jetson Nano) because it doesn't connect from the Microsfot RDP...
Then add the following line at the end of the file:
>lxsession -s LXDE -e LXDE
## 6. Test your connection using Microsoft RDP by entering the ip address of the device and then the credentials.

 <br />  

# IV. Install Python packages(optional).
## 1. Make sure you have Python installed also pip too.
* `python --version`
* `python3 --version`
* `pip --version`

## 2. Install Python development and its packages.
```
sudo apt-get install python-pip
sudo apt-get install python python-dev
``` 
or
```
sudo apt-get install python3-pip
sudo apt-get install python3 python3-dev
```

## 3. Ensure apt-get is up to date.
* `sudo apt-get update` (upgrade is optional since in Jetson it may crash the system).

## 4. Run the top level CMake command but this time with an additional flag -DBUILD_PYTHON_BINDINGS:bool=true.
**Make sure that you are in the root directory.**
* `mkdir build`
* `cd build`
* `cmake ../ -DBUILD_PYTHON_BINDINGS:bool=true -DPYTHON_EXECUTABLE=/usr/bin/python3 -DCMAKE_BUILD_TYPE=release -DBUILD_EXAMPLES=true -DBUILD_GRAPHICAL_EXAMPLES=true`

* `make -j4` *(4 is number of CPU cores to parallelize the build process)*
* `sudo make install`

## 5. Update PYTHONPATH env. variable to add the path to the pyrealsense library.
* `export PYTHONPATH=$PYTHONPATH:/usr/local/lib`

## 6. If this doesn't work do the steps from: https://github.com/IntelRealSense/librealsense/issues/6964.
 <br />  

 <br />  

# V. SLAM with D435i.
## 1. Install the ROS distribution.
Since on the Jetson Nano runs Ubuntu 18.04 we must install the melodic distribution from: http://wiki.ros.org/melodic/Installation/Ubuntu.

### 1.1 Setup your sources.list
* `sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'`

### 1.2 Set up your keys
* `sudo apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654`

### 1.3 Installation
First make sure that your packages are up to date:
```
sudo apt update
export ROS_VER=melodic
sudo apt-get install ros-$ROS_VER-realsense2-camera
sudo apt install ros-melodic-desktop-full 
```
This will install both realsense2_camera and its dependents, including librealsense2 library.

### 1.4 Check for available packages
```
apt search ros-melodic
```

### 1.5 Setup your environment
```
echo "source /opt/ros/melodic/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

### 1.6 Dependencies for building packages.
To install this tool and other dependencies for building ROS packages, run:
```
sudo apt install python-rosdep python-rosinstall python-rosinstall-generator python-wstool build-essential
```
### 1.7 Initialize rosdep
```
sudo rosdep init
rosdep update
```

sudo apt install python-rosdep python-rosinstall python-rosinstall-generator python-wstool build-essential
## 2. Install Intel® RealSense™ ROS from Sources
### 2.1 Create a catkin workspace Ubuntu**
```
mkdir -p ~/catkin_ws/src
cd ~/catkin_ws/src/
```
### 2.2 Clone the latest Intel® RealSense™ ROS from https://github.com/IntelRealSense/realsense-ros/releases into 'catkin_ws/src/'
```
git clone https://github.com/ros-perception/vision_opencv
git clone https://github.com/IntelRealSense/realsense-ros.git
cd realsense-ros/
git checkout `git tag | sort -V | grep -P "^2.\d+\.\d+" | tail -1`
cd .. 
```

## After the jetson hacks tutorial
```
sudo apt install ros-melodic-rtabmap-ros
sudo apt install ros-melodic-robot-localization
sudo apt-get install ros-melodic-imu-tools
```

## In case of error like this
```
libGL error: MESA-LOADER: failed to open swrast (search paths /usr/lib/aarch64-linux-gnu/dri:\$${ORIGIN}/dri:/usr/lib/dri)
libGL error: failed to load driver: swrast
Could not open OpenGL window, please check your graphic drivers or use the textual SDK tools
```

You need to create a soft link between some libdrm libraries:
```
cd /usr/lib/aarch64-linux-gnu
sudo ln -sf libdrm.so.2.4.0 libdrm.so.2
```

## Versions
* librealsense SDK v2.40.0
* realsense-viewer v2.40.0
* Firmware version: 05.12.09.00
## Kernel
* Linux 4.9.140-tegra #1 SMP PREEMPT Fri Oct 16 12:32:46 PDT 2020 aarch64 aarch64 aarch64 GNU/Linux \
Distributor ID:	Ubuntu \
Description:	Ubuntu 18.04.5 LTS \
Release:	18.04 \
Codename:	bionic 

## Compilers
gcc (Ubuntu/Linaro 7.5.0-3ubuntu1~18.04) 7.5.0 \
cmake version 3.10.2 \
opencv 4.1.1

## IDE
1.52.1 ea3859d4ba2f3e577a159bc91e3074c5d85c0523 arm64


