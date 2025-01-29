Gaze Control is a python based script that uses your webcam to enable you to control your PC using only your nose and eyes.It takes the spatial position of your eyes and maps it to your Windows cursor.As for the eyes when you blink both your eyes once it registers as a left click and on blinking twice consecutively registers as a right click.

GazeControl is mainly oriented towards differently abled users.Improving their quality of life and ease of life.Being an early release it is suceptible to having a few bugs.Such as the incorrect blink detection when looking up or down and the script not being able to detect blinks due to very strong light source behind the user.

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
