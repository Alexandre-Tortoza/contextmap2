# FAST-LIO (ROS 1 Noetic) image used by the FAST-LIO state estimation deployment.
#
# Build recipe of the image `contextual-3d-mapping-fastlio-ros1`, whose local build
# (image ID sha256:670973462caa9496acaf41dd2a94f07277f14439b4d026d3bd9965c81b1dd985,
# built 2026-09-09) ran the reference validation of the FAST-LIO backend. The three
# source refs are pinned: the build is reproducible from them. The image holds only the
# built FAST-LIO; the deployment wrapper (src/contextmap/state_estimation/backends/
# fast_lio_wrapper.py) is mounted at run time, so changing it never needs a rebuild.
#
#   docker build -t contextmap-fast-lio-ros1 docker/fast-lio-ros1
#
# FAST-LIO is GPL-2.0 (hku-mars/FAST_LIO); it is built and run here, not redistributed.
FROM osrf/ros:noetic-desktop-full

SHELL ["/bin/bash", "-c"]
ARG FAST_LIO_REF=7cc4175de6f8ba2edf34bab02a42195b141027e9
ARG LIVOX_SDK_REF=9306596a2bf15c1343bc023b497465ed0a32909d
ARG LIVOX_ROS_DRIVER_REF=3d240d5666129e1a3052e78ee8487a04b08fdda3

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential cmake git libapr1-dev \
    && rm -rf /var/lib/apt/lists/*

# FAST-LIO's messages and CMake depend on the Livox driver even for other LiDAR types.
RUN git clone https://github.com/Livox-SDK/Livox-SDK.git /tmp/Livox-SDK \
    && git -C /tmp/Livox-SDK checkout --detach "${LIVOX_SDK_REF}" \
    && cmake -S /tmp/Livox-SDK -B /tmp/Livox-SDK/build -DCMAKE_BUILD_TYPE=Release \
    && cmake --build /tmp/Livox-SDK/build --parallel \
    && cmake --install /tmp/Livox-SDK/build \
    && rm -rf /tmp/Livox-SDK

RUN mkdir -p /opt/fast-lio/src \
    && git clone https://github.com/Livox-SDK/livox_ros_driver.git /opt/fast-lio/src/livox_ros_driver \
    && git -C /opt/fast-lio/src/livox_ros_driver checkout --detach "${LIVOX_ROS_DRIVER_REF}" \
    && git clone https://github.com/hku-mars/FAST_LIO.git /opt/fast-lio/src/FAST_LIO \
    && git -C /opt/fast-lio/src/FAST_LIO checkout --detach "${FAST_LIO_REF}" \
    && git -C /opt/fast-lio/src/FAST_LIO submodule update --init --recursive

RUN source /opt/ros/noetic/setup.bash \
    && cd /opt/fast-lio \
    && catkin_make -DCMAKE_BUILD_TYPE=Release

WORKDIR /workspace
CMD ["bash"]
