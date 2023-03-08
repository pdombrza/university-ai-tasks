import json
import numpy as np
import gymnasium as gym
from parameters import *


class Agent:
    def __init__(self, learning_rate: float, n_states: int, n_actions: int, discount: float, init_temp: float, final_temp: float, temp_decay: float) -> None:
        self.learning_rate = learning_rate
        self.q_values = np.zeros(shape=(n_states, n_actions))
        self.discount = discount
        self.temperature = init_temp
        self.temp_decay = temp_decay
        self.final_temp = final_temp

    def choose_action_boltzmann(self, state):
        # choose next action - based on boltzmann strategy
        prob_numenator = (np.exp(self.q_values[state]) / self.temperature)
        numenator_sum = sum(prob_numenator)
        probabilities = prob_numenator / numenator_sum
        actions_array = np.arange(self.q_values.shape[1])
        action = np.random.choice(actions_array, p=probabilities)
        return action

    def choose_action_epsilon(self, state):
        epsilon = self.temperature
        if np.random.uniform() < epsilon:
            action = np.random.choice(np.arange(self.q_values.shape[1]))
        else:
            action = np.argmax(self.q_values[state])
        return action

    def choose_action_best(self, state):
        return np.argmax(self.q_values[state])

    def update(self, state, action, reward, next_state) -> None:
        # update knowledge of map
        current_qval = self.q_values[state][action]
        diff = reward + self.discount * np.max(self.q_values[next_state]) - current_qval
        new_qval = current_qval + self.learning_rate * diff
        self.q_values[state][action] = new_qval

    def decay_temperature(self) -> None:
        self.temperature = max(self.final_temp, self.temperature - self.temp_decay)


def qlearn(agent: Agent, episode_count: int, env: gym.Env, choice_func: callable, eval: bool=False, decay: bool=False) -> list:
    reward_arr = []
    episode_lengths = []
    for _ in range(episode_count):
        state, info = env.reset()
        terminated = False
        truncated = False
        reward_sum = 0
        episode_length = 0
        while not truncated and not terminated:
            episode_length += 1
            action = choice_func(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            reward_sum += reward
            if not eval:
                agent.update(state, action, reward, next_state)
            state = next_state
        reward_arr.append(reward_sum)
        episode_lengths.append(episode_length)
        if not eval and decay:
            agent.decay_temperature()
    return reward_arr, episode_lengths


def evaluate(agent: Agent, test_episode_count: int, env: gym.Env, choice_func: callable) -> list:
    rewards, _ = qlearn(agent, test_episode_count, env, choice_func, eval=True)
    return rewards


def visualize(env: gym.Env, agent: Agent) -> None:
    env = gym.make('Taxi-v3', render_mode="human")
    evaluate(agent, 20, env, agent.choose_action_best)


def save_data(fname: str, data: dict) -> None:
    with open(fname, 'w') as fh:
        json.dump(data, fh)


def learn_and_evaluate(env: gym.Env, agent: Agent, choice_func: callable, max_episodes: int, learning_period: int, eval_period: int, decay: bool=False) -> tuple:
    avg_rewards = []
    training_rewards = []
    training_episode_lengths = []
    for i in range(max_episodes // learning_period):
        r, l = qlearn(agent, learning_period, env, choice_func, decay=decay)
        training_rewards.append(r)
        training_episode_lengths.append(l)
        rewards = evaluate(agent, eval_period, env, agent.choose_action_best)
        avg_rewards.append(sum(rewards) / eval_period)
    print("Finished training. ")
    return avg_rewards, training_rewards, training_episode_lengths


def main():
    env = gym.make('Taxi-v3')
    env.reset()
    taxi = Agent(LEARNING_RATE, env.observation_space.n, env.action_space.n, DISCOUNT, START_EPSILON, FINAL_TEMPERATURE, EPSILON_DECAY)
    avg_rewards, training_rewards, training_episode_lengths = learn_and_evaluate(env, taxi, taxi.choose_action_boltzmann, MAX_EPISODES, LEARNING_PERIOD, EVAL_PERIOD, decay=False)

    # visualize(env, agent=taxi)
    data = {"avg_rewards": avg_rewards, "training_rewards": training_rewards, "training_episode_lengths": training_episode_lengths}
    print(avg_rewards)
    # save_data("results_epsilon.json", data)


if __name__ == "__main__":
    main()
