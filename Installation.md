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
    * `cmake ../ -DBUILD_EXAMPLES=true` - Builds *librealsense* along with the demos and tutorials<br />

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

 <br />  
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
* `sudo apt-get install python python-dev` or `sudo apt-get install python3 python3-dev`

## 3. Ensure apt-get is up to date.
* `sudo apt-get update` (upgrade is optional since in Jetson it may crash the system).

## 4. Run the top level CMake command but this time with an additional flag -DBUILD_PYTHON_BINDINGS:bool=true.
**Make sure that you are in the root directory.**
* `mkdir build`
* `cd build`
* `cmake ../-DFORCE_RSUSB_BACKEND=ON -DBUILD_PYTHON_BINDINGS:bool=true -DPYTHON_EXECUTABLE=/usr/bin/python3 -DCMAKE_BUILD_TYPE=release -DBUILD_EXAMPLES=true -DBUILD_GRAPHICAL_EXAMPLES=true -DBUILD_WITH_CUDA:bool=true`

* `make -j4` *(4 is number of CPU cores to parallelize the build process)*
* `sudo make install`

## 5. Update PYTHONPATH env. variable to add the path to the pyrealsense library.
* `export PYTHONPATH=$PYTHONPATH>/usr/local/lib`

## 6. If this doesn't work do the steps from: https://github.com/IntelRealSense/librealsense/issues/6964.
 <br />  