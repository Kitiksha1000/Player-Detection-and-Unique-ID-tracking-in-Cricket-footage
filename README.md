# Player Detection and Unique ID Tracking in Cricket Footage

## 1. Project Overview
This computer vision pipeline is designed to detect players in cricket match footage and assign consistent, unique IDs to them across the video sequence. It leverages **YOLOv5** for object detection and **DeepSORT** for multi-object tracking (MOT). The solution addresses challenges such as occlusion, visual similarity (uniforms), and camera motion.

**Key Features:**
* **Robust Tracking:** Uses Kalman Filters and Re-Identification (ReID) to maintain IDs even after temporary occlusions.
* **Crash Recovery:** Implements a chunk-wise processing system that saves progress automatically to Google Drive, allowing the script to resume after runtime disconnects.
* **Resource Optimized:** Designed to run within the constraints of the Google Colab free tier (Tesla T4 GPU).

---

## 2. Dependencies
The project requires a Python environment with GPU support (CUDA). The specific libraries used are:

* **Python 3.8+**
* **YOLOv5 (Ultralytics)**: For object detection.
* **DeepSORT Realtime**: For tracking algorithms (Kalman Filter + ReID).
* **OpenCV (`opencv-python-headless`)**: For video frame manipulation.
* **PyTorch**: Deep learning framework backend.
* **FFmpeg**: For lossless video merging.

**Automatic Installation:**
The notebook includes a setup cell that automatically installs the required non-standard libraries

## Setup Instructions

### 1. Prepare Your Environment

1. Upload the `.ipynb` notebook file to Google Colab
2. Upload your cricket video footage to Google Drive (e.g., `Khel_AI_assignment_Video_1.mp4`)
3. Create an output folder in Google Drive for processed chunks

### 2. Mount Google Drive

Run the drive mounting cell to enable file access:
```python
from google.colab import drive
drive.mount('/content/drive')
```

### 3. Configure Paths

Update the configuration cell with your specific paths:
```python
INPUT_VIDEO_PATH = '/content/drive/MyDrive/your_folder/Khel_AI_assignment_Video_1.mp4'
OUTPUT_DIR = '/content/drive/MyDrive/your_output_folder/'
```

## Running the Pipeline

Execute the notebook cells in sequential order:

### Step 1: Configuration
- Locate and run the configuration cell
- Verify `INPUT_VIDEO_PATH` points to your video file
- Ensure `OUTPUT_DIR` is set to a valid Google Drive folder

### Step 2: Model Loading
- Run initialization cells to download YOLOv5 weights
- Initialize the DeepSORT tracker

### Step 3: Video Processing
- Run the main processing loop
- The pipeline will process video in chunks and save progress automatically
- **Note**: If Colab crashes or times out, simply restart the runtime and re-run the processing cell - it will automatically resume from the last saved chunk

### Step 4: Final Merge
- Run the merging cells to combine all chunks
- The script copies chunks locally to prevent Drive I/O errors
- FFmpeg concatenates chunks into a single seamless video

## Output

The pipeline generates:

### Intermediate Files
- **Chunk Files**: Individual `.mp4` files saved to Google Drive
  - Format: `tracked_part_00001.mp4`, `tracked_part_00002.mp4`, etc.
  - Automatically saved during processing for crash recovery

### Final Output
- **Merged Video**: Single complete file
  - Filename: `final_merged_output.mp4` or `final_tracked_output.mp4`
  - Location: Specified output directory

## Troubleshooting

### Common Issues

#### Google Colab Timeouts
- **Symptom**: Runtime disconnects during long videos (45+ minutes)
- **Solution**: Built-in resume logic handles this automatically
  1. Restart the runtime
  2. Re-run the processing cell
  3. Pipeline detects existing chunks and continues from last saved frame

#### Drive Latency/File Corruption
- **Symptom**: Errors during video merging
- **Solution**: Merging step performs local copy (`shutil.copy`) before processing
- **Note**: This is intentional to avoid corruption from reading streams directly from Google Drive

### Performance Optimization

- Process videos in smaller segments if experiencing frequent timeouts
- Ensure stable internet connection for Google Drive synchronization
- Monitor GPU usage in Colab to stay within free tier limits

