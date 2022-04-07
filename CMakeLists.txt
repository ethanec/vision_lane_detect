cmake_minimum_required(VERSION 2.8.3)
project(vision_lane_detect)


find_package(autoware_msgs REQUIRED)
find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  image_transport
  roscpp
  autoware_msgs
	tf # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!Fucking Important
	message_generation
	####### YC ####### 
	vector_map_msgs 
	dbw_mkz_msgs
	rospy
	####### YC ####### 
)
find_package(OpenCV REQUIRED)
find_package(OpenGL REQUIRED)

####### YC ####### 
catkin_python_setup()
####### YC #######
	 
add_message_files(
	FILES
 	Float32MultiArray_H.msg
	Int32MultiArray_H.msg
)
generate_messages(         
	DEPENDENCIES
	std_msgs
)

catkin_package(
  #  INCLUDE_DIRS include
  #  LIBRARIES vision_lane_detect
  CATKIN_DEPENDS roscpp message_runtime
  #  DEPENDS system_lib
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

###########
## Build ##
###########

include_directories(
        ${catkin_INCLUDE_DIRS}
        ${autoware_msgs_INCLUDE_DIRS}
        nodes/vision_lane_detect
				nodes/HD_lane_detect
				nodes/HD_lane_detect_opt
        ${OpenCV_INCLUDE_DIRS}
)


add_executable(vision_lane_detect
        nodes/vision_lane_detect/vision_lane_detect.cpp
        )

add_executable(HD_lane_detect
        nodes/HD_lane_detect/HD_lane_detect.cpp
        )

add_executable(HD_lane_detect_opt
        nodes/HD_lane_detect_opt/HD_lane_detect_opt.cpp
        )


target_link_libraries(vision_lane_detect
        ${catkin_LIBRARIES}
        ${OpenCV_LIBRARIES}
        ${OPENGL_LIBRARIES}
        )

target_link_libraries(HD_lane_detect
        ${catkin_LIBRARIES}
        ${OpenCV_LIBRARIES}
        ${OPENGL_LIBRARIES}
        )

target_link_libraries(HD_lane_detect_opt
        ${catkin_LIBRARIES}
        ${OpenCV_LIBRARIES}
        ${OPENGL_LIBRARIES}
        )


add_dependencies(vision_lane_detect
        ${catkin_EXPORTED_TARGETS}
        )

add_dependencies(HD_lane_detect
        ${catkin_EXPORTED_TARGETS}
        )

add_dependencies(HD_lane_detect_opt
        ${catkin_EXPORTED_TARGETS}
        )


install(TARGETS vision_lane_detect
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(TARGETS HD_lane_detect
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(TARGETS HD_lane_detect_opt
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)







