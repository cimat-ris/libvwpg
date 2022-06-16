cmake_minimum_required(VERSION 2.8...3.22)

project (libvwpg)

############################################################
add_compile_options(-Wno-dev -Wno-ignored-attributes -Wno-deprecated-declarations  -D__USE_LONG_INTEGERS__)

find_package(OpenCV REQUIRED PATHS "/usr/lib/x86_64-linux-gnu/cmake/opencv4/")
find_library(qpOASES NAMES libqpOASES PATHS qpOASES/bin)
set (CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/lib/")
set (EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

############################################################
# Include directories
############################################################
find_path(EIGEN_INCLUDE_DIRS "Eigen/Core" PATHS "${SDK_DIR}/include" PATH_SUFFIXES "eigen3")
if (EIGEN_INCLUDE_DIRS)
    include_directories ("${EIGEN_INCLUDE_DIRS}")
endif (EIGEN_INCLUDE_DIRS)
include_directories ("${PROJECT_SOURCE_DIR}/include")
include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories(/usr/include/eigen3 qpOASES/include include)

set (libvwpg_SRC "${CMAKE_CURRENT_SOURCE_DIR}/./src/common.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/./src/feet.cpp"
	     	"${CMAKE_CURRENT_SOURCE_DIR}/./src/herdt.cpp"
	     	"${CMAKE_CURRENT_SOURCE_DIR}/./src/qp.cpp"
	     	"${CMAKE_CURRENT_SOURCE_DIR}/./src/vision_utils.cpp"
	        "${CMAKE_CURRENT_SOURCE_DIR}/./src/homography.cpp"
	        "${CMAKE_CURRENT_SOURCE_DIR}/./src/essential.cpp"
	        "${CMAKE_CURRENT_SOURCE_DIR}/./src/factory.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/./src/visualfeaturebase.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/./src/models.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/./src/trunk_orientation.cpp"
	      	"${CMAKE_CURRENT_SOURCE_DIR}/./src/common_formulation.cpp"
			"${CMAKE_CURRENT_SOURCE_DIR}/./src/QPSimulators.cpp")

link_directories(qpOASES/bin/)

############################################################
# Create the dynamic library
############################################################
add_library(libvwpg SHARED ${libvwpg_SRC})
target_link_libraries(libvwpg opencv_core opencv_calib3d qpOASES)

############################################################
# Create the test1 executable
############################################################
add_executable(test1-vwpg  "${CMAKE_CURRENT_SOURCE_DIR}/./src/test1.cpp")
target_link_libraries(test1-vwpg libvwpg opencv_core opencv_calib3d qpOASES)

############################################################
# Create the test2 executable
############################################################
# qi_create_bin(test2-vwpg  "${CMAKE_CURRENT_SOURCE_DIR}/./src/test2.cpp")
# qi_use_lib(test2-vwpg boost_filesystem boost_system qpoases OPENCV2_CORE OPENCV2_HIGHGUI)
# target_link_libraries(test2-vwpg libvwpg)

############################################################
# Create  python binding
############################################################
# OPTION (BUILD_PYTHON_INTERFACE "Build the python binding" ON)
# IF(BUILD_PYTHON_INTERFACE)
#substitute your version number
#find_package(Boost 1.58 EXACT REQUIRED COMPONENTS python)
# find_package(Boost 1.55 EXACT REQUIRED COMPONENTS python)
# INCLUDE_DIRECTORIES("${Boost_INCLUDE_DIRS}" "/usr/include/python2.7" " qpOASES/include")
# find_package( PythonInterp 2.7 REQUIRED )
# find_package( PythonLibs 2.7 REQUIRED )


############################################################
# Create the test1 executable
############################################################
# PYTHON_ADD_MODULE(QPHomograpy_py src/QPHomograpy.cpp)
#_use_lib(QPHomograpy_py boost_filesystem boost_system qpoases OPENCV2_CORE OPENCV2_HIGHGUI)
#target_link_libraries(QPHomograpy_py libvwpg ${PYTHON_LIBRARIES} ${Boost_LIBRARIES})
#  INSTALL(TARGETS QPHomograpy_py DESTINATION ${PYTHON_INSTALL_DIR})
#ENDIF(BUILD_PYTHON_INTERFACE)
