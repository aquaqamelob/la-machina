---
language:
- en
- fr
license: mit
tags:
- yolo26
- computer-vision
- object-detection
- sports
- football
- soccer
- tracking
library_name: ultralytics
pipeline_tag: object-detection
---

# Football Match Analysis

## Model Description

This project uses **YOLO v26** models for real-time football match analysis, including:

- Player detection and tracking
- Team classification by jersey color
- Referee detection (orange/yellow/pink jerseys)
- Formation line visualization
- Possession and pass statistics

## Intended Use

- Sports analytics and coaching
- Match analysis and review
- Training video annotation
- Research in sports computer vision

## Models Used

| Model | Purpose | Size |
|-------|---------|------|
| `yolo26m.pt` | Player/ball detection | ~50MB |
| `yolo26m-pose.pt` | Pose estimation (optional) | ~50MB |

Download models from [Ultralytics](https://docs.ultralytics.com/models/yolo26/).

## How to Use

```python
# Clone the repository
git clone https://github.com/YOUR_USERNAME/football-analysis.git
cd football-analysis

# Install dependencies
pip install -r requirements.txt

# Download YOLO models
# Models are automatically downloaded on first run

# Run analysis
python main.py input_video.mp4 output_video.mp4
```

## Features

### Team Classification
- Automatic team color detection using K-means clustering
- HSV-based referee detection (orange, yellow, pink jerseys)
- Temporal smoothing to avoid flickering

### Formation Lines
- Detects horizontal player alignments (3+ players)
- Maximum 2 connections per player
- Visualizes defensive/midfield/attack lines

### Statistics
- Real-time possession percentage
- Pass count per team
- Footer overlay with live stats

## Configuration

Key parameters in `config.py`:

```python
detection_confidence = 0.3  # Player detection threshold
ball_confidence = 0.2       # Ball detection threshold
device = "mps"              # Auto-detected: mps/cuda/cpu
```

## Limitations

- Requires clear camera angle (broadcast view works best)
- Teams must have distinct jersey colors
- Possession stats are approximate (based on ball proximity)
- Performance varies with video quality

## Performance

| Device | FPS |
|--------|-----|
| Mac M4 (MPS) | 15-20 |
| NVIDIA RTX 3080 | 25-30 |
| CPU (i7) | 3-5 |

## Training Data

This project uses pre-trained YOLO v26 models from Ultralytics, trained on:
- COCO dataset (person, sports ball classes)
- No additional fine-tuning on football-specific data

## Ethical Considerations

- This tool is intended for sports analysis only
- Should not be used for surveillance purposes
- Respect privacy when analyzing non-public matches

## Citation

```bibtex
@software{football_analysis,
  title={Football Match Analysis with YOLO v26},
  year={2026},
  url={https://github.com/David-dsv/football-analysis}
}
```

## License

MIT License
# la-machina
# la-machina
