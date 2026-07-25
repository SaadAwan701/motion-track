# Real-Time Motion Detection & Centroid Tracking

A lightweight, real-time computer vision pipeline built with Python and OpenCV.

This project was built as a hands-on foundation to master core computer vision concepts - moving beyond black-box deep learning models to understand classical spatial processing, frame differencing, morphological operations, and object tracking algorithms.

---

## Key Features & Vision Concepts Learned

- **Frame Differencing & Background Adaptation** - Detects moving objects using absolute frame subtraction (`|Frame_A - Frame_B|`) with dynamic baseline updates to handle ambient light shifts.
- **Image Preprocessing** - Grayscale conversion and Gaussian blurring to suppress high-frequency camera sensor noise.
- **Morphological Processing** - Binary thresholding and matrix dilation to form clean spatial masks from scattered contour points.
- **Centroid Multi-Object Tracking** - Assigns persistent unique IDs (`ID 0`, `ID 1`, ...) using Euclidean spatial distance matching (`np.linalg.norm`) across consecutive frame states.
- **State Lifecycle Management** - Tracks active centroids and automatically deregisters objects after missing for a configurable frame threshold (`max_disappeared`).
- **Automated Unit Test Suite** - Fully tested using `pytest` and synthetic NumPy image matrices.

---

## Project Architecture

```
motion-track/
├── motion_track/
│   ├── __init__.py
│   ├── detector.py         # Core MotionDetector class (differencing & contours)
│   └── tracker.py          # CentroidTracker class (spatial distance ID matching)
├── tests/
│   ├── test_detector.py    # Unit tests for motion detection on synthetic matrices
│   └── test_tracker.py     # Unit tests for tracker state lifecycle management
├── app.py                  # Main CLI application entry point
├── pytest.ini               # Test discovery configuration
└── requirements.txt        # Dependency specifications
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- Webcam or an input video file (`.mp4`, `.avi`)

### Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/YOUR_USERNAME/motion-track.git
   cd motion-track
   ```

2. **Set up a virtual environment:**

   ```bash
   # Windows (PowerShell)
   python -m venv venv
   .\venv\Scripts\Activate.ps1

   # macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### Run Live Feed with Webcam

```bash
python app.py
```

### Run on a Pre-Recorded Video File

```bash
python app.py --video path/to/sample_video.mp4
```

**Controls:** Press `q` while focused on the video window to stop the application and release hardware resources.

---

## Running the Unit Tests

Automated tests evaluate core pipeline components against controlled synthetic image matrices without requiring a physical camera stream.

To run the complete test suite:

```bash
pytest
```

---

## License

Distributed under the MIT License. See [`LICENSE`](LICENSE) for details.
