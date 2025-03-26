# Use the official Ubuntu 20.04 base image
FROM nvidia/cudagl:11.3.0-devel-ubuntu20.04

# Update the package lists and install Python 3.8, pip, and other dependencies
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3.8 python3-pip git libopenexr-dev ffmpeg libturbojpeg tmux wget libxcb-xinerama0 libqt5gui5 && \
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
