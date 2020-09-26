# robot-sac
- Repo for CS 8903 Special Problems Course at Georgia Tech.
- Implementation of [Soft Actor-Critic algorithm by Haarnoja et. al.](https://arxiv.org/abs/1801.01290) and [Deep Deterministic Policy Gradients by Lillicrap et. al.](https://arxiv.org/pdf/1509.02971.pdf) 
- Implemented using Swift for Tensorflow, Tested on Open AI: Gym environments

# Deep Deterministic Policy Gradients
- [DDPG Paper:](https://arxiv.org/pdf/1509.02971.pdf) 
- The implementation of this algorithm can be found in ddpg.swift. This script contains code for the Actor and Critic networks and also includes the training
  setup for the DDPG algorithm.
- To run this script simply run "swift ddpg.swift"  This script will train a DDPG agent on the inverted pendulum problem from [gym](https://gym.openai.com/envs/Pendulum-v0/)
- I also made a notebook on Google Colab with this same code:[Link to Notebook](https://colab.research.google.com/drive/1Lmf-CVubsPRhPmcfJ3Dc-gpWXVnF3dcZ?usp=sharing)
