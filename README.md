# MarIO-Kart - Mario Kart 64 AI Using Supervised and Reinforcement Learning
Reinforcement Learning on N64 Mario Kart

## Set-up


### Install Python Dependencies
The following Python dependencies need to be installed.

- tensorflow
- gym
- gym_mupen64Plus
- mupen64plus
- Pillow
- scikit-image

### Mupen64Plus and Gym
First, install mupgen64plus emulator (via apt-get) and install gym via ```pip install gym```. After gym is installed, follow https://github.com/bzier/gym-mupen64plus repository to install all dependencies and input bots for setting up the environment between OpenAI's Gym and mupen64plus meulator.  

### Recording and Viewing Data 
Instructions to record samples used were based off of and slightly modified from Hudges (https://github.com/kevinhughes27/TensorKart) TensorKart project.

## Usage Instructions
1) After recording samples, run ```python utilities.py prepare /path/to/recordings/``` which will turn all recorded samples into X.npy and y.npy files.  
2) ```config.py``` contains all parameters needed to run the script, such as save directory, where weights are saved. Modify these configurations to suit your environement.  
3) Run ```python train_MarIO.py``` which takes parameters ```-s``` or ```-dqn```. These specify whether the network is to undergo Deep Q Reinforcement Learning or Supervised Learning. Ensure that the data directory in config.py matches where processed data samples are.
4) After training, you may run ```python play_MarIO.py``` with arguments ```-s``` or ```-dqn``` to specify which network to use.

## Special Thanks

- [TensorKart](https://github.com/kevinhughes27/TensorKart) - The first MarioKart deep learning project
  [Hughes 2017] HUGHES, Kevin: TensorKart. In: GitHub repository (2017). – URL https://github.com/kevinhughes27/TensorKart 
- [Atari with Reinforcement Learning] 
  [Mnih u. a. 2013] MNIH, Volodymyr ; KAVUKCUOGLU, Koray ; SILVER, David ; GRAVES, Alex ;
  ANTONOGLOU, Ioannis ; WIERSTRA, Daan ; RIEDMILLER, Martin: Playing Atari With Deep
  Reinforcement Learning. (2013)
