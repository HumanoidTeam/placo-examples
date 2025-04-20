cmake_minimum_required(VERSION 3.22)
project(prealpha_description)

find_package(ament_cmake REQUIRED)

install(DIRECTORY meshes urdf control rviz launch
        DESTINATION share/${PROJECT_NAME}/
)

ament_package()
