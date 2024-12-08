# docker build . -t aloha_real -f src/openpi/runtime/environments/aloha_real/Dockerfile
# docker run --rm -it --network=host -v /dev:/dev -v .:/root/app --privileged aloha_real /bin/bash

FROM ros:noetic-robot@sha256:0e12e4db836e78c74c4b04c6d16f185d9a18d2b13cf5580747efa075eb6dc6e0
SHELL ["/bin/bash", "-c"]

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    cmake \
    curl \
    libffi-dev \
    python3-rosdep \
    python3-rosinstall \
    python3-rosinstall-generator \
    whiptail \
    git \
    wget \
    openssh-client \
    ros-noetic-cv-bridge \
    ros-noetic-usb-cam \
    ros-noetic-realsense2-camera \
    keyboard-configuration

WORKDIR /root
RUN curl 'https://raw.githubusercontent.com/Interbotix/interbotix_ros_manipulators/main/interbotix_ros_xsarms/install/amd64/xsarm_amd64_install.sh' > xsarm_amd64_install.sh
RUN chmod +x xsarm_amd64_install.sh
RUN export TZ='America/Los_Angeles' && ./xsarm_amd64_install.sh -d noetic -n

COPY ./third_party/aloha /root/interbotix_ws/src/aloha
COPY ./docker/aloha_act/arm.py /root/interbotix_ws/src/interbotix_ros_toolboxes/interbotix_xs_toolbox/interbotix_xs_modules/src/interbotix_xs_modules/arm.py
RUN cd /root/interbotix_ws && source /opt/ros/noetic/setup.sh && source /root/interbotix_ws/devel/setup.sh && catkin_make

# Install python 3.10 because this ROS image comes with 3.8
RUN mkdir /python && \
    cd /python && \
    wget https://www.python.org/ftp/python/3.10.14/Python-3.10.14.tgz && \
    tar -zxvf Python-3.10.14.tgz && \
    cd Python-3.10.14 && \
    ls -lhR && \
    ./configure --enable-optimizations && \
    make install && \
    echo 'alias python3="/usr/local/bin/python3.10"' >> ~/.bashrc && \
    echo 'alias python="/usr/local/bin/python3.10"' >> ~/.bashrc && \
    cd ~ && rm -rf /python && \
    rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.5.6 /uv /bin/uv
ENV UV_HTTP_TIMEOUT=120
ENV UV_LINK_MODE=copy
COPY ./src/openpi/runtime/environments/requirements_ros1.txt /tmp/requirements.txt
RUN uv pip sync --python 3.10 --system /tmp/requirements.txt

ENV PYTHONPATH=$PYTHONPATH:/root/interbotix_ws/src/aloha/aloha_scripts:/root/interbotix_ws/src/aloha:/root/app/src
WORKDIR /root/app

# Create an entrypoint script to run the setup commands, followed by the command passed in.
RUN cat <<'EOF' > /usr/local/bin/entrypoint.sh
#!/bin/bash
source /opt/ros/noetic/setup.sh && source /root/interbotix_ws/devel/setup.sh && "$@"
EOF
RUN chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["python3", "/root/app/src/openpi/runtime/environments/aloha_real/main.py"]
