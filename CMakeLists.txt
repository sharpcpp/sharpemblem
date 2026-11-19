cmake_minimum_required(VERSION 3.25)

project(sharpemblem LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set (OpenCV_DIR "C:/opencv/opencv/build") # Set the path to your OpenCV build directory

find_package(OpenCV CONFIG REQUIRED
    COMPONENTS core imgproc imgcodecs
)

set(SHARPEMBLEM_SRC_DIR ${PROJECT_SOURCE_DIR}/src)

set(SHARPEMBLEM_SRC_BF4 ${SHARPEMBLEM_SRC_DIR}/bf4/*.h ${SHARPEMBLEM_SRC_DIR}/bf4/*.cpp)

set(SHARPEMBLEM_SRC ${SHARPEMBLEM_SRC_DIR}/*.h ${SHARPEMBLEM_SRC_DIR}/*.cpp)

set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/output)

set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/output)

file(GLOB SHARPEMBLEM_SOURCES_BF4 ${SHARPEMBLEM_SRC_BF4})

file(GLOB SHARPEMBLEM_SOURCES ${SHARPEMBLEM_SRC})

add_executable(sharpemblem ${SHARPEMBLEM_SOURCES} ${SHARPEMBLEM_SOURCES_BF4})

target_include_directories(sharpemblem PRIVATE ${SHARPEMBLEM_SRC_DIR})

target_link_libraries(sharpemblem PRIVATE
    ${OpenCV_LIBS}
)
set(JSON_BuildTests OFF CACHE INTERNAL "")
set(JSON_Install OFF CACHE INTERNAL "")

add_subdirectory(external/json)

target_link_libraries(sharpemblem
    PRIVATE
        nlohmann_json::nlohmann_json
)

target_compile_options(sharpemblem PRIVATE -mavx2)

target_include_directories(sharpemblem PRIVATE
    ${OpenCV_INCLUDE_DIRS}
)