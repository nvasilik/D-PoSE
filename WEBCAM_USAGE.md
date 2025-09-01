# D-PoSE ROS2 Webcam Demo - Usage Instructions

This cleaned version of `ros_demo_webcam.py` provides an easy-to-use interface for real-time 3D human pose estimation using a webcam.

## Requirements

Before running the script, ensure you have:

1. **Docker** installed and properly configured
2. **CUDA-capable GPU** for optimal performance
5. **NVIDIA Container Toolkit** installed
3. **Webcam** connected to your system
4. **D-PoSE model files**:
   - `data/ckpt/paper_arxiv.ckpt`

## Basic Usage

### Build Image and run a container
```bash
./run_dpose.sh
```

### Simple Usage (with defaults)
```bash
python3 ros_demo_webcam.py
```

### Common Usage Examples

```bash
# Use external USB camera (camera ID 1)
python3 ros_demo_webcam.py --camera-id 1

# Enable live video display window
python3 ros_demo_webcam.py --display

# Use different camera resolution
python3 ros_demo_webcam.py --width 1920 --height 1080

# Enable ArUco marker detection for camera calibration
python3 ros_demo_webcam.py --use-aruco

# Use custom model configuration
python3 ros_demo_webcam.py --cfg my_config.yaml --ckpt my_model.ckpt

# Lower detection threshold for more sensitive detection
python3 ros_demo_webcam.py --detection-threshold 0.5

# Combine multiple options
python3 ros_demo_webcam.py --camera-id 1 --display --use-aruco --fps 30
```

## Command Line Options

### Essential Options
- `--camera-id`: Camera device ID (default: 0)
- `--display`: Show live video window (use 'q' to quit)
- `--cfg`: Path to model configuration file
- `--ckpt`: Path to model checkpoint file

### Camera Configuration
- `--width`: Camera width in pixels (default: 1280)
- `--height`: Camera height in pixels (default: 720)  
- `--fps`: Frame rate (default: 13)

### Processing Options
- `--detection-threshold`: Person detection confidence (default: 0.7)
- `--detector`: Detector type - 'yolo' or 'maskrcnn' (default: maskrcnn)

### Additional Features
- `--use-aruco`: Enable ArUco marker detection
- `--output-folder`: Directory for log files (default: ./logs)

## Getting Help

```bash
# View all available options with descriptions
python3 ros_demo_webcam.py --help
```

## ROS2 Topics Published

The script publishes to the following ROS2 topics:

- `/humans` (skeleton_msgs/Skeletons): 3D skeleton data for all detected persons
- **TF Transforms**: Joint positions as transforms for visualization in RViz

## Troubleshooting

### Common Issues

**Camera not found:**
```
Error: Cannot open camera 0
```
Solution: Try different camera IDs (--camera-id 1, 2, etc.) or check camera connections.

**CUDA not available:**
```
Warning: CUDA not available, using CPU (will be slower)
```
Solution: Install CUDA drivers and PyTorch with CUDA support, or accept slower CPU processing.

**Model files not found:**
```
Required file not found: data/ckpt/paper_arxiv.ckpt
```
Solution: Download the model checkpoint as described in the main README.md

**ROS2 not configured:**
```
ModuleNotFoundError: No module named 'rclpy'
```
Solution: Source your ROS2 setup: `source /opt/ros/humble/setup.bash`

### Performance Tips

1. **Use CUDA**: Ensure PyTorch is installed with CUDA support
2. **Adjust resolution**: Lower camera resolution for better FPS
3. **Adjust detection threshold**: Higher threshold = fewer false positives but may miss some people
4. **Close video display**: Disable `--display` for better performance in production

## Safety Notes

- Press 'q' in the video window to quit safely
- Use Ctrl+C in terminal as backup to stop the program
- Camera LED will turn off when program exits properly

## Integration with RViz

To visualize the pose estimation results:

1. Launch RViz: `rviz2`
2. Add TF display to see joint transforms
3. Set fixed frame to "Camera" or "base_link"
4. The script publishes transforms for all detected human joints
