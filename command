cd ws_livox
source /opt/ros/noetic/setup.bash
catkin_make
source devel/setup.bash
roslaunch cloud2voxel_mapping cloud2voxel_mapping.launch