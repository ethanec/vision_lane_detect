* Autoware should be launch first!
	* At the **Map** tab, launch the **Vector Map**
	* At the **Sensing** tab, launch the **Calibration Publisher**

* Go to $vision_lane_detect root, inside the `script/`, run `python lane_net.py`.
* Go to Autoware/ros/src, run `rosrun vision_lane_detect HD_lane_detect`.
* Show the results in RVIZ.
