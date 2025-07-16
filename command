cd ws_livox
source /opt/ros/noetic/setup.bash
catkin_make
source devel/setup.bash
roslaunch cloud2voxel_mapping cloud2voxel_mapping.launch

SIBR_viewers\bin\SIBR_gaussianViewer_app_rwdi.exe -m output
