# vision_lane_detect

## Launching process
1. Autoware should be launch first!
	- At the **Map** tab, launch the **Vector Map**
	- At the **Sensing** tab, launch the **Calibration Publisher**

2. Go to $vision_lane_detect root, inside the `script/`, run `python lane_net.py`.
3. Go to Autoware/ros/src, run `rosrun vision_lane_detect HD_lane_detect`.
4. Show the results in RVIZ.

# Rebuild environment on Autoware 1.13
It's just able to run sucessfully on Ubuntu 18.04 with the launch file, maybe it will fail when operating other files

### needed files
- vision_lane_detect package
- conputing.yaml
- Calibration_20200918_v4_best.yml

### modified part
- vision_lane_detect/CMakeLists.txt
    ```=.txt
    find_package(catkin REQUIRED COMPONENTS
        ####### YC ####### 
        vector_map_msgs 
        dbw_mkz_msgs
        rospy
        ####### YC ####### 
    )

    ####### YC ####### 
    catkin_python_setup()
    ####### YC #######

    catkin_package(
        ####### YC ####### 
        vector_map_msgs 
        dbw_mkz_msgs
        ####### YC #######
    )
    
    ####### YC ####### 

    catkin_install_python(PROGRAMS
        script/self_localization/KF_lane_self_localization_opt_run.py
        DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
    )
    ####### YC ####### 
    ```
    
- vision_lane_detect/package.xml
    ```=.xml
            <!-- YC -->
		<build_depend>vector_map_msgs</build_depend>
		<build_depend>dbw_mkz_msgs</build_depend>
		<!-- YC -->
        
            <!-- YC -->
		<run_depend>vector_map_msgs</run_depend>
		<run_depend>dbw_mkz_msgs</run_depend>
		<!-- YC -->
    ```
- add file `setup.py` under`vision_lane_detect`, which ables the python file to import package `utils`
    ```=python
    from setuptools import setup
    from catkin_pkg.python_setup import generate_distutils_setup

    setup_args = generate_distutils_setup(
        #version="1.10.0",
        packages=['utils'],
        package_dir={'': 'script/self_localization'})

    setup(**setup_args)
    ```
- complie autoware workspace
