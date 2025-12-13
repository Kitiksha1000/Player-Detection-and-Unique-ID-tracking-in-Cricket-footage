# Player Tracking Pipeline using YOLOv8 and ByteTrack

**Author:** Kitiksha Chakrawarti

## Project Overview
This project implements a computer vision pipeline for detecting and tracking players in sports videos (specifically football/soccer). It utilizes **YOLOv8** for state-of-the-art object detection and **ByteTrack** for robust multi-object tracking. The system processes video footage to identify players, assign unique tracking IDs, and visualize their movement trajectories.

The solution is designed to be run in a GPU-accelerated environment (specifically Google Colab) and includes necessary workarounds for library compatibility between recent YOLO versions and the ByteTrack algorithm.

## Dependencies
The project requires **Python 3.x** and a GPU-enabled environment. Key dependencies include:

* **Ultralytics YOLOv8** (`ultralytics<=8.3.40`): For player detection.
* **ByteTrack**: Multi-object tracking algorithm (installed from source).
* **Supervision** (`supervision==0.1.0`): For annotating video frames and handling geometry.
* **YOLOX**: A dependency required by ByteTrack.
* **NumPy**: Specific version compatibility required for YOLOX.
* **Helper Libraries**: `cython_bbox`, `onemetric`, `loguru`, `lap`, `thop`, `opencv-python`.

## Installation Instructions

### Option 1: Google Colab (Recommended)
This notebook is optimized for Google Colab. No manual local setup is required.
1.  Open the notebook in Google Colab.
2.  Run the **Environment Setup** cells at the very top. These cells will automatically:
    * Install the required Python packages (`ultralytics`, `supervision`, etc.).
    * Clone the ByteTrack repository.
    * Install ByteTrack dependencies and apply necessary patches.

### Option 2: Local Installation
If you prefer to run this locally, ensure you have a CUDA-capable GPU and follow these steps:

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/ifzhang/ByteTrack.git](https://github.com/ifzhang/ByteTrack.git)
    cd ByteTrack
    ```

2.  **Install Python Dependencies**
    ```bash
    pip install "ultralytics<=8.3.40" supervision==0.1.0
    pip install cython_bbox onemetric loguru lap thop
    ```

3.  **Install ByteTrack**
    Inside the `ByteTrack` directory:
    ```bash
    pip install -r requirements.txt
    python3 setup.py develop
    ```

## Steps to Run the Pipeline

1.  **Enable GPU Acceleration**
    * **Colab:** Go to `Edit` > `Notebook settings` > Select `GPU` (e.g., T4) as the Hardware accelerator.
    * **Local:** Ensure your CUDA drivers are correctly installed.

2.  **Mount Storage (Colab Only)**
    * Run the cell to mount Google Drive:
        ```python
        from google.colab import drive
        drive.mount('/content/drive')
        ```
    * Ensure your input video is uploaded to your Google Drive.

3.  **Run Setup and Imports**
    * Execute the initialization cells to load libraries and define the `BYTETrackerArgs` class.

4.  **Configure Paths**
    * Locate the variable `SOURCE_VIDEO_PATH` in the notebook.
    * Update it to point to your specific video file (e.g., `/content/drive/MyDrive/path/to/video.mp4`).

5.  **Execute Tracking**
    * Run the main inference loop.
    * The pipeline will:
        1.  Extract frames from the video.
        2.  Detect players using YOLOv8 (`yolov8x.pt` is used by default).
        3.  Track players across frames using ByteTrack.
        4.  Annotate the video with bounding boxes and IDs.
        5.  Save the resulting video to the output path.
