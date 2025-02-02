
# Simultaneous Localization and Mapping Using Fiducial Markers
Travis:  [![Build Status](https://travis-ci.org/UbiquityRobotics/fiducials.svg?branch=kinetic-devel)](https://travis-ci.org/UbiquityRobotics/fiducials)

Jenkins: [![Build Status](http://build.ros.org/view/Kdev/job/Kdev__fiducials__ubuntu_xenial_amd64/badge/icon)](http://build.ros.org/view/Kdev/job/Kdev__fiducials__ubuntu_xenial_amd64/)

## Overview

This package implements a system that uses ceiling mounted
fiducial markers (think QR Codes) to allow a robot to identify
its location and orientation.  It does this by constructing
a map of the ceiling fiducials.  The position of one fiducial
needs to be specified, then a map of the fiducials is built 
up from observing pairs of markers in the same image. 
Once the map has been constructed, the robot can identify
its location by locating itself relative to one or more 
ceiling fiducials.

Documentation is at [http://wiki.ros.org/fiducials](http://wiki.ros.org/fiducials).

## Recording A Bag File

Sometimes for trobleshooting purposes it is useful to record a bag 
file to capture the exact data on the topics going into and out of 
fiducials.

To do this, while the system is running, run `rosbag record -a`.
You can upload this bag file to a file sharing service like Google
Drive and link to it in your issue, this will help us diagnose 
the problem. 

## 희준 변경 사항
Error 없앰

`src/fiducials/aruco_detect/scripts/axis_base_to_cam.py ` 여기에 Eligen base to cam tf를 계산해놓음

### 1. boost의 Library Link가 되지 않은 문제!!

```bash
/usr/bin/ld: CMakeFiles/fiducial_slam.dir/src/map.cpp.o: in function `boost::filesystem::path::parent_path() const':
map.cpp:(.text._ZNK5boost10filesystem4path11parent_pathEv[_ZNK5boost10filesystem4path11parent_pathEv]+0x2c): undefined reference to `boost::filesystem::path::find_parent_path_size() const'
collect2: error: ld returned 1 exit status
make[2]: *** [fiducials/fiducial_slam/CMakeFiles/fiducial_slam.dir/build.make:203: /home/inspacehj/ros_projects/eligen_test_ws/devel/lib/fiducial_slam/fiducial_slam] Error 1
make[1]: *** [CMakeFiles/Makefile2:2778: fiducials/fiducial_slam/CMakeFiles/fiducial_slam.dir/all] Error 2
```

[Boost cmake error 해결 한 링크](https://iostream1029.tistory.com/23) 를 따라가서 문제가 생긴 `fiducial_slam/CMakeLists.txt`에 가서 수정해줌

```Cmake
find_package(OpenCV REQUIRED)

# added
find_package(Boost COMPONENTS filesystem)
...

# ${Boost_LIBRARIES} added
target_link_libraries(fiducial_slam ${catkin_LIBRARIES} ${OpenCV_LIBS} ${Boost_LIBRARIES})

```
### 2. launch file이 작동되지 않는 이슈..
```bash
❯ roslaunch aruco_detect rs_aruco_detect.launch
... logging to /home/inspacehj/.ros/log/e372e7f8-624b-11ec-8132-0050080d15ba/roslaunch-Inspaceui-Heejun-Jang-84868.log
Checking log directory for disk usage. This may take a while.
Press Ctrl-C to interrupt
Done checking log file disk usage. Usage is <1GB.

started roslaunch server http://hj2004:41725/

SUMMARY
========

PARAMETERS
 * /aruco_detect/dictionary: 16
 * /aruco_detect/do_pose_estimation: True
 * /aruco_detect/fiducial_len: 0.18
 * /aruco_detect/fiducial_len_override: 
 * /aruco_detect/ignore_fiducials: 
 * /aruco_detect/publish_images: True
 * /aruco_detect/verbose: False
 * /aruco_detect/vis_msgs: True
 * /rosdistro: noetic
 * /rosversion: 1.15.13

NODES
  /
    aruco_detect (aruco_detect/aruco_detect)

ROS_MASTER_URI=http://hccbip:11311/

process[aruco_detect-1]: started with pid [85003]
[aruco_detect-1] process has died [pid 85003, exit code -11, cmd /home/inspacehj/ros_projects/eligen_test_ws/devel/lib/aruco_detect/aruco_detect camera/compressed:=/rs_cam/color/image_raw/compressed camera_info:=/rs_cam/color/camera_info __name:=aruco_detect __log:=/home/inspacehj/.ros/log/e372e7f8-624b-11ec-8132-0050080d15ba/aruco_detect-1.log].
log file: /home/inspacehj/.ros/log/e372e7f8-624b-11ec-8132-0050080d15ba/aruco_detect-1*.log
all processes on machine have died, roslaunch will exit
shutting down processing monitor...
... shutting down processing monitor complete
done
```
이 문제는 혹시 몰라 aruco_detect 폴더 CMakeLists.txt OPENCV 버전 부분 수정

`find_package(OpenCV REQUIRED)` -> `find_package(OpenCV 4.2.0 REQUIRED)` 

이러니 되네!

아마도.. OpenCV 3.2로 빌드해서 생긴 문제인듯 함... **Fxxk**