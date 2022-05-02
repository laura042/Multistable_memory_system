# Multistable memory system

The aim of this project is to simulate a 1D multistable chain composed of coupled bistable spring-mass units and to control
it using a Reinforcement Learning agent, Twin-Delayed DDPG. We used the TD3 agent implemented here [PFRL](https://github.com/pfnet/pfrl).
This repository was used to produce the results of the article "Dynamically writing coupled memories using a reinforcement learning agent, meeting physical bounds".

### Description of the repository

The repertory [gym_systmemoire](gym_systmemoire) contains the environment simulating the multistable chain. 
The file [Config_env.py](Config_env.py) is used to configure the environment.
The repertory [pfrl-master](pfrl-master) contains the RL agent and functions to train it. This repertory can be found here [PFRL](https://github.com/pfnet/pfrl).
In this project, we have modified the file [train_agent.py](pfrl-master/pfrl/experiments/train_agent.py).
The file [Train_phase.py](Train_phase.py) is used to train the agent and the file [Test_phase.py](Test_phase.py) generates a chosen number of episodes or steps to test 
the learned models. 
The repertory [TL](TL) is used to do Transfer Learning from a regime to others by varying the friction coefficient (see fig. 2 b) of the article).
The repertory [two_internal_time_scales](two_internal_time_scales) is used to generate the data of fig. 3 and the repertory [scaling_analysis](scaling_analysis) is used to generate the 
data of fig. 4.


### Installation

The environment can be installed using :

`python setup.py install`

PFRL can be installed using :

`cd pfrl-master`

`python setup.py install`
