# GazeControl

GazeControl is a Python-based application that leverages your webcam to enable hands-free PC control using facial tracking powered by MediaPipe and PyTorch. The script maps the spatial position of your nose to your Windows cursor, allowing for smooth navigation. Additionally, eye blinks serve as input gestures—a single blink triggers a left-click, while two consecutive blinks register as a right-click.

Designed primarily for differently-abled users, GazeControl aims to enhance accessibility and improve ease of use. As an early release, it may exhibit occasional issues, such as inaccurate blink detection when the user looks up or down, or failure to detect blinks in environments with intense backlighting.

This implementation utilizes CUDA acceleration (if available) for real-time tracking and integrates smoothing algorithms for cursor stability. Future updates will focus on refining detection accuracy and expanding functionality.

# Pre-requisite

Python

Pytorch

CUDA(Optional)

# Installation 
### Clone the repository and install requirements.txt
```
git clone https://github.com/silentstudios0725/GazeControl
cd GazeControl
pip install requirements.txt
```

### Install PyTorch 

go to https://pytorch.org/get-started/locally/ for Pytorch According to your config

### With CPU
```
pip3 install torch torchvision torchaudio
```
### With CUDA
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```
