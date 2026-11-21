FROM ros:humble-ros-base-jammy

SHELL ["/bin/bash", "-c"]

ENV DEBIAN_FRONTEND=noninteractive
ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
ENV GO1_MODELS_DIR=/opt/mujoco_menagerie/unitree_go1


RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ca-certificates \
    build-essential \
    python3-pip \
    python3-colcon-common-extensions \
    ros-humble-rmw-cyclonedds-cpp \
    ros-humble-rosidl-default-generators \
    ros-humble-rosidl-generator-dds-idl \
    && rm -rf /var/lib/apt/lists/*


RUN git init /tmp/unitree_ros2 && cd /tmp/unitree_ros2 \
    && git remote add origin https://github.com/unitreerobotics/unitree_ros2.git \
    && git sparse-checkout set cyclonedds_ws/src/unitree/unitree_go \
    && git fetch --depth 1 origin 668d1ec5a05d1c38d3306bdca7d59f2ba3581a88 \
    && git checkout FETCH_HEAD \
    && mkdir -p /opt/unitree_ws/src \
    && mv cyclonedds_ws/src/unitree/unitree_go /opt/unitree_ws/src/unitree_go \
    && rm -rf /tmp/unitree_ros2

WORKDIR /opt/unitree_ws
RUN source /opt/ros/humble/setup.bash \
    && colcon build --packages-select unitree_go \
    && rm -rf build log
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt


COPY models/unitree_go1/ /opt/mujoco_menagerie/unitree_go1/


RUN source /opt/ros/humble/setup.bash \
    && source /opt/unitree_ws/install/setup.bash \
    && python3 -c "from unitree_go.msg import LowCmd, LowState; import mujoco; m = mujoco.MjModel.from_xml_path('/opt/mujoco_menagerie/unitree_go1/scene.xml'); assert m.nu == 12"


COPY entrypoint.sh /entrypoint.sh
RUN chmod 755 /entrypoint.sh

RUN useradd -m -s /bin/bash participant
COPY safety_block.py /home/participant/safety_block.py
USER participant
WORKDIR /home/participant

ENTRYPOINT ["/entrypoint.sh"]
