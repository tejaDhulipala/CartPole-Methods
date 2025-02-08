import gymnasium as gym
import pygame as pg
import numpy as np
import torch
from CrossEntropyAgent import CrossEntropyAgent
from DeepQAgent import QAgent

agent = None
if input("Choose Q for DQN, A for Cross-Entropy: ") == "Q":
    agent = QAgent(4, 2)
    agent.q_network.load_state_dict(torch.load("q_network.pth"))
    agent.q_target.load_state_dict(torch.load("q_network.pth"))
else:
    agent = CrossEntropyAgent(4, 2)
    agent.policy_network.load_state_dict(torch.load("cross_entropy_network.pth"))

env = gym.make("CartPole-v1", render_mode="human")
pg.init()
observation, info = env.reset()

episode_over = False
agent_return = 0
while not episode_over:
    action = agent.action(observation) if type(agent) == CrossEntropyAgent else agent.action(observation, 0)
    observation, reward, terminated, truncated, info = env.step(action)
    agent_return += reward

    env.render()
    for event in pg.event.get():
        if event.type == pg.QUIT:
            episode_over = True
        if event.type == pg.KEYDOWN:
            if pg.key.get_pressed()[pg.K_r]:
                observation, info = env.reset()
    if terminated:
        print(agent_return)
        agent_return = 0
        observation, info = env.reset()

env.close()