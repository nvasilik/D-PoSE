# Base image with CUDA + OpenGL on Ubuntu 20.04
FROM nvidia/cudagl:11.3.0-devel-ubuntu20.04

# Install system dependencies, Python, pip, and cleanup
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3 python3-pip git libopenexr-dev ffmpeg libturbojpeg tmux wget \
    libxcb-xinerama0 libqt5gui5 \
    && python3 -m pip install --upgrade --no-cache-dir pip setuptools \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy ROS setup script and install ROS
COPY setup_ros.sh .
RUN chmod +x setup_ros.sh && ./setup_ros.sh
WORKDIR /app/dpose
#source /opt/ros/rolling/setup.bash 
RUN echo "source /opt/ros/rolling/setup.bash" >> ~/.bashrc

WORKDIR /app/dpose/ros2_ws
RUN colcon build
RUN echo "source /app/dpose/ros2_ws/install/setup.bash" >> ~/.bashrc
WORKDIR /app/dpose

# Default command
CMD ["/bin/bash"]
