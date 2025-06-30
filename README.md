# Real-Time EMG-EEG-Based BCI: Random Dot Motion Decision Task

## Overview

This repository contains Python code for a real-time closed-loop Brain-Computer Interface (BCI) experiment investigating the "point of no return" in sensorimotor decision-making. The experiment implements a Random Dot Motion (RDM) task with evidence termination paradigm to test the double integration framework in motor control.

## Research Objectives

The experiment aims to determine:
1. When a motor action becomes irreversible after initial motor preparation begins
2. The minimum time required for an EMG burst to trigger a completed action
3. Whether changing perceptual evidence after EMG onset can affect action completion

## Key Features

- **Real-time EMG monitoring** with <10ms latency for onset detection
- **Continuous Random Dot Motion** stimulus presentation to avoid visual evoked potentials
- **Closed-loop paradigm** where visual evidence terminates based on EMG activity
- **Precise timing control** with hardware-accelerated PsychoPy implementation
- **Lab Streaming Layer (LSL)** integration for EMG-PsychoPy communication
- **Adaptive delay mechanisms** (10, 20, 30ms increments) between EMG onset and evidence termination

## Technical Specifications

### Hardware Requirements
- High-refresh monitor (60Hz minimum, 120Hz recommended)
- EMG recording system with real-time processing (1000Hz+ sampling)
- EEG system (optional, for offline analysis)
- Response buttons/keypads
- Lab Streaming Layer (LSL) compatible setup

### Stimulus Parameters
- **Dot Motion**: 16.7 dots/degree², 2-3 pixel dots, 5°/s speed, 50ms lifetime
- **Coherence Levels**: 50% (high) and 0% (termination)
- **Aperture**: 7.1° diameter circular field
- **Fixation**: Continuous 5×5 pixel white square (0.5° visual angle)
- **Frame Rate**: 60fps minimum

### Trial Structure
- **ITI**: Randomly varied (3, 6, 9 seconds)
- **High Coherence Phase**: Up to 1.5 seconds
- **Evidence Termination**: Triggered by EMG onset + variable delay (Δt)
- **Confidence Rating**: 1-7 scale (post-response)

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/emg-eeg-bci-experiment.git
cd emg-eeg-bci-experiment

# Install Python dependencies
pip install -r requirements.txt

# Required packages include:
# - psychopy>=2023.1.0
# - numpy>=1.21.0
# - pylsl>=1.16.0
# - scipy>=1.8.0
# - matplotlib>=3.5.0
```


### Configuration Options
```bash
# Test EMG connection
python src/emg_interface.py --test

# Calibrate timing precision
python tests/timing_validation.py

# Run practice session
python src/main_experiment.py --participant P001 --practice
```

## Data Output

The experiment generates several data files:
- **Behavioral data**: Response times, accuracy, confidence ratings
- **EMG data**: Onset times, burst characteristics, response completion
- **Stimulus timing**: Coherence change timestamps, frame timing validation
- **Trial metadata**: Condition parameters, trial outcomes

## Key Implementation Features

### Real-Time EMG Processing
- High-pass filtering (10Hz cutoff) with full-wave rectification
- Threshold detection (2-3 SD above baseline)
- Maximum 10ms latency from onset to PsychoPy trigger

### Timing Precision
- Hardware-accelerated stimulus presentation
- Frame drop monitoring and trial exclusion
- Photodiode validation for critical timing events

### Closed-Loop Control
- Seamless coherence transitions (50% → 0%) without VEPs
- Variable EMG-to-termination delays (Δt: 10, 20, 30ms)
- Continuous fixation maintenance throughout trials

## Experimental Design

- **10-12 blocks** of 60-80 trials each
- **Counterbalanced** motion directions (left/right)
- **Systematic variation** of EMG-coherence delays across blocks
- **Adaptive procedures** to focus on individual "point of no return"

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/emg-optimization`)
3. Test timing accuracy with provided validation tools
4. Commit changes with detailed descriptions
5. Submit pull request with performance metrics


## Contact

[Hamed Ghane] - [hamed.ghanesasansaraei@glasgow.ac.uk]
Project Link: https://github.com/HamedGhanes/Emg-EEG-BCI
Project Lab: https://mphiliastides.org/
## Acknowledgments

- Lab Streaming Layer (LSL) community for real-time data streaming
- PsychoPy development team for precise stimulus control
