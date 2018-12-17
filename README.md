[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135623-e770e354-7d12-11e8-998d-29fc74429ca2.gif "Trained Agent"
[image2]: https://user-images.githubusercontent.com/10624937/42135622-e55fb586-7d12-11e8-8a54-3c31da15a90a.gif "Soccer"


# Project 3: Collaboration and Competition

### Introduction

For this project, we work with the [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment.

![Trained Agent][image1]

In this environment, two agents control rackets to bounce a ball over a net. 
If an agent hits the ball over the net, it receives a reward of **+0.1**. 
If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of **-0.01**.
Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. 
The state space is in total of dimension **24**. Each agent receives its own, local observation.  
Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping. 
The action space is hence of size **2**.

The task is episodic, and in order to solve the environment, the agents must get an average score of **+0.5** 
(over **100** consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single **score** for each episode.

The environment is considered solved, when the average (over 100 episodes) of those **scores** is at least **+0.5**.

### Getting Started

1. Clone this repository, which contains all files to install dependencies, start the environment on Windows (64-bit) 
computer and (re)train the agents:
    - [Tennis_Windows_x86_64](https://github.com/schambon77/DRLND-Collaboration-and-Competition/tree/master/Tennis_Windows_x86_64):
    Tennis Unity enviroment for Windows 64-bit
    - Source code
        - [main.py](https://github.com/schambon77/DRLND-Collaboration-and-Competition/blob/master/main.py): main script
        - [maddpg.py](https://github.com/schambon77/DRLND-Collaboration-and-Competition/blob/master/maddpg.py)
        - [ddpg.py](https://github.com/schambon77/DRLND-Collaboration-and-Competition/blob/master/ddpg.py)
        - [buffer.py](https://github.com/schambon77/DRLND-Collaboration-and-Competition/blob/master/buffer.py)
        - [networkforall.py](https://github.com/schambon77/DRLND-Collaboration-and-Competition/blob/master/networkforall.py)
        - [OUNoise.py](https://github.com/schambon77/DRLND-Collaboration-and-Competition/blob/master/OUNoise.py)
        - [utilities.py](https://github.com/schambon77/DRLND-Collaboration-and-Competition/blob/master/utilities.py)
    - [checkpoint_actor_0.pth](https://github.com/schambon77/DRLND-Collaboration-and-Competition/blob/master/checkpoint_actor_0.pthh): 
    saved trained actor 0 neural network coefficients
    - [checkpoint_actor_1.pth](https://github.com/schambon77/DRLND-Collaboration-and-Competition/blob/master/checkpoint_actor_1.pthh): 
    saved trained actor 1 neural network coefficients
    - [checkpoint_critic.pth](https://github.com/schambon77/DRLND-Collaboration-and-Competition/blob/master/checkpoint_critic.pth): 
    saved trained shared critic neural network coefficients
    - [report.md](https://github.com/schambon77/DRLND-Collaboration-and-Competition/blob/master/report.md): report
 
2. If not using a Windows (64-bit) computer, download the environment from one of the links below. 
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)

3. Follow instructions in the [instructions in the DRLND GitHub repository](https://github.com/udacity/deep-reinforcement-learning#dependencies) 
to set up your Python environment

### Instructions

To run the code to train the agents, run the [main.py](https://github.com/schambon77/DRLND-Collaboration-and-Competition/blob/master/main.py) script.

### Report

The [report](https://github.com/schambon77/DRLND-Collaboration-and-Competition/blob/master/report.md) provides a description of the implementation.
