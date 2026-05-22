# EVLS TrueLock Pipeline v2.2

Complete pipeline for consistent AI character generation using **TrueLock** methodology.

## Overview

This project provides a robust system for extracting 3D landmarks, performing visual analysis, and enforcing strict character consistency across image generations.

**Main Character:** Alexandra Marchis (Petite Hourglass - Bottom-Heavy)

## Skills Included

| Skill                  | Purpose                                      | Key Features |
|------------------------|----------------------------------------------|--------------|
| `landmark-provider`    | MediaPipe landmark extraction                | Face + Pose World Landmarks (3D meters), profile mode support |
| `vision-analyzer-v6`   | 3D visual analysis & metrics                 | Body proportions, sizes in cm, IPD, emotional state detection |
| `truelock`             | Consistency enforcement                      | Deviation scoring, TrueLock validation |

## Quick Start

### Installation

```bash
pip install -e .
```

Or install dependencies manually:

```bash
pip install -r requirements.txt
```

## Key Features

- Real 3D coordinates from MediaPipe World Landmarks
- Height-based scaling for accurate cm measurements
- Facial expression & emotional state detection
- Side-profile face detection support
- Strict TrueLock consistency enforcement

## License

MIT