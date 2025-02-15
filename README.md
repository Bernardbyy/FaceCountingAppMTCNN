# FaceCountingAppMTCNN

This project aims to develop an advanced face detection system that combines multiple computer vision techniques to improve the accuracy of face detection and counting.

## Core Components

### 1. Base Technology
MTCNN (Multi-Task Cascaded Convolutional Neural Networks) serves as the foundation for detecting faces. MTCNN is a popular deep learning model for face detection.

### 2. Enhancement Pipeline
The project improves MTCNN's performance by adding three main stages of image processing:

- **Image Preprocessing**
  - CLAHE (Contrast Limited Adaptive Histogram Equalization)
  - Gaussian Blur
  - Unsharp Masking for image quality enhancement

- **Segmentation**
  - DeepLabV3 for separating people from backgrounds

- **Feature Extraction**
  - Skin tone detection for better facial region identification

### 3. User Interface
A PyQt5-based GUI that provides:

- Multiple input sources:
  - Webcam
  - Clipboard
  - Local files

- Real-time controls:
  - Parameter adjustment for processing techniques
  - Face detection visualization with bounding boxes
  - Live face count display
  - Performance monitoring over time
