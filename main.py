import gymnasium as gym
import pygame as pg
import numpy as np
import torch
from Cross_Entropy.CrossEntropyAgent import CrossEntropyAgent
from Q_Learning.DeepQAgent import QAgent
from REINFORCE.ReinforceAgent import ReinforceAgent
from Agent import Agent

agent = None
match input("Choose Q for DQN, C for Cross-Entropy, R for reinforce, H for human: ").upper():
    case "Q":
        agent = QAgent(4, 2)
        agent.q_network.load_state_dict(torch.load("q_network.pth"))
        agent.q_target.load_state_dict(torch.load("q_network.pth"))
    case "C":
        agent = CrossEntropyAgent(4, 2)
        agent.policy_network.load_state_dict(torch.load("cross_entropy_network.pth"))
    case "R":
        agent = ReinforceAgent(4, 2)
        agent.policy_network.load_state_dict(torch.load("reinforce.pth"))
    case "H":
        pass
    case default:
        agent = Agent(4, 2)

env = gym.make("CartPole-v1", render_mode="human")
pg.init()
observation, info = env.reset()

episode_over = False
agent_return = 0
action = 0
while not episode_over:
    for event in pg.event.get():
        if event.type == pg.QUIT:
            episode_over = True
        if event.type == pg.KEYDOWN:               
            if pg.key.get_pressed()[pg.K_r]:
                observation, info = env.reset()
    
    if not agent == None: 
        action = agent.action(observation, 0) if type(agent) == QAgent else agent.action(observation)
    else:
        if pg.key.get_pressed()[pg.K_RIGHT]:
            action = 1
        else:
            action = 0
    observation, reward, terminated, truncated, info = env.step(action)
    agent_return += reward

    if terminated and not agent == None:
        print(agent_return)
        agent_return = 0
        observation, info = env.reset()
    
    env.render()

env.close()