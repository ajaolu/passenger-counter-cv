# Real-Time Automated Passenger Counting System

A computer vision pipeline engineered to detect, track, and count passengers navigating transit points or facility entryways using deep learning. 

## 🚀 Key Features
- **Object Detection:** Optimized detection framework tailored for identifying passengers in video streams.
- **Directional Tracking:** Smart line-crossing logic to count entries and exits while eliminating duplicate counts.
- **Modular Architecture:** Clean separation between the core computer vision detection loop and tracking analytics.

## 🛠️ System Architecture & Workflow
1. **Input Stream:** Accepts static video files or live RTSP IP camera streams.
2. **Detection Engine:** Identifies objects frame-by-frame using deep learning.
3. **Tracking & Analytics:** Assigns persistent IDs to tracking bounding boxes and increments counts upon boundary line intersections.

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.10 or higher
- `pip` (Python package manager)

### Step-by-Step Installation
1. Clone this repository to your local machine:
   ```bash
   git clone [https://github.com/ajaolu/passenger-counting-cv.git](https://github.com/ajaolu/passenger-counting-cv.git)
   cd passenger-counting-cv
## 📺 Video Demo
[Watch the system in action here:](https://www.facebook.com/share/r/1E7MKDBH1d/)
