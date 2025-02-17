import gymnasium as gym
from Agent import Agent
import numpy as np

class Recorder:
    def __init__(self, env: gym.Env, agent: Agent, discount=1):
        self.env = env
        self.recording = []
        self.agent = agent
        self.discount_factor = discount

    def record_one_episode(self, last_episode=True, custom_reward = False, episode_limit=-1):
        """
        Records one episode

        Episode limit is the maximum number of episodes that can be run. If it's -1, it'll stop when truncated.

        Returns a tuple in the form:
        (array of [(state, action, reward)], total return)
        """
        running = True
        observation, info = self.env.reset()
        episode = []
        return_value = 0
        discount = 1
        while running:
            action = self.agent.action(observation)
            t_step = [observation, action]
            observation, reward, terminated, truncated, info = self.env.step(action)
            if custom_reward:
                reward = 0.418 - abs(observation[2])
            t_step = tuple(t_step + [reward])
            episode.append(t_step)
            return_value += reward * discount
            discount *= self.discount_factor

            if terminated:
                running = False
            if (len(episode) > episode_limit and episode_limit > 0) or (episode_limit < 0 and truncated):
                print("EPISODE TRUNCATED")
                running = False
        
        if last_episode:
            self.env.close()

        return (episode, return_value)

    def record_episodes(self, N: int, custom_reward=False, epsode_limit=-1):
        """
        records N episodes to self.recording
        """
        for i in range(N):
            self.recording.append(self.record_one_episode(last_episode=(i==N-1), custom_reward=custom_reward, episode_limit=epsode_limit))
    
    def reset(self):
        self.__init__(self.env, self.agent, discount=self.discount_factor)
    
    def sort_episodes(self):
        """
        changes the recording array such that the episodes are arranged in decreasing order
        """
        rewards = np.array(list(zip(*self.recording))[1])
        order = np.argsort(rewards)
        new_recording = []
        for i in range(len(self.recording)):
            new_recording.append(self.recording[order[i]])
        self.recording = new_recording   
    
    def percentile_episodes(self, percentile, is_destructive=False):
        """
        percentile: The top "percentile" of episodes to be considered
        if destrucitive is true, self.recording gets changed. Otherwise, it doesn't
        Either way it returns the array of episode tuples 
        """
        start = int(len(self.recording) * (1 - percentile / 100))
        top_episodes = self.recording[start:]
        if is_destructive:
            self.recording = top_episodes
        return top_episodes