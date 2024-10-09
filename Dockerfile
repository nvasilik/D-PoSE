# Use the official Ubuntu 20.04 base image
FROM ubuntu:20.04

# Update the package lists and install Python 3.8, pip, and other dependencies
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3.8 python3-pip git libopenexr-dev libgl1 ffmpeg libsm6 libxext6 libglfw3-dev libgles2-mesa-dev libturbojpeg tmux wget && \
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get -y install cuda-toolkit-12-4 && \
    rm cuda-keyring_1.1-1_all.deb && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip and setuptools
RUN python3.8 -m pip install --upgrade pip setuptools

# Set the working directory
WORKDIR /app

# Copy the requirements.txt file to the working directory
COPY requirements.txt .

# Install Python dependencies from requirements.txt
RUN pip install -r requirements.txt

# Set the command to launch a bash shell
CMD ["/bin/bash"]
