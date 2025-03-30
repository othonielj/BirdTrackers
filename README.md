# BirdTrackers

A Python-based system for tracking and analyzing bird movements in video footage using computer vision and deep learning techniques.

## Features

- Bird detection using OpenCV and deep learning models
- Multi-object tracking with deep learning-based trackers
- Bird movement analysis and visualization
- Support for various video formats
- Configurable tracking parameters

## Requirements

- Python 3.8+
- OpenCV
- NumPy
- Deep learning frameworks (as specified in requirements.txt)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/BirdTrackers.git
cd BirdTrackers
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Place your video files in the `data/input` directory
2. Run the detection and tracking pipeline:
```bash
python scripts/A_detect_with_cvlib.py
python scripts/B_filter_detections.py
python scripts/C_track_birds.py
python scripts/D_deep_track_birds.py
python scripts/E_compare_trackers.py
```

3. View results in the `data/output` directory

## Project Structure

```
BirdTrackers/
├── data/
│   ├── input/          # Input video files
│   └── output/         # Generated results
├── scripts/            # Python scripts
├── requirements.txt    # Project dependencies
└── README.md          # This file
```

## License

[Your chosen license]

## Contributing

[Your contribution guidelines] 