import numpy as np

class SARSA:
    def __init__(self, num_actions, num_states, gamma=1.0, initial_alpha=0.50, initial_epsilon=0.20):
        self.A = num_actions
        self.S = num_states
        self.Q = np.zeros(dtype=np.float32, shape=[self.S, self.A])

        self.gamma = gamma

        self.initial_alpha = initial_alpha
        self.initial_epsilon = initial_epsilon
        self.alpha = self.initial_alpha
        self.epsilon = self.initial_epsilon

        self.steps = 0
        self.episodes = 0

    def act(self, s):
        if np.random.uniform(0.0, 1.0) > self.epsilon:
            # greedy
            a = np.argmax(self.Q[s,:])
            return a
        else:
            # exploring
            a = np.random.randint(0, self.A)
            return a

    def learn(self, s, a, r, s_prime, a_prime, episode_done):
        '''
        :param s: int, state_id for which an action was executed in the environment
        :param a: int, action_id for action executed in the environment
        :param r: float, reward received for executing action a in state s, and transitioning to state s_prime.
        :param s_prime: int, state_id for the state transitioned to after executing action a
        :param a_prime: int, a hypothetical next action to execute.
        :param episode_done: boolean indicating whether the episode is done. required for GLIE schedule.
        :return: None
        '''
        Q_next = self.Q[s_prime, a_prime] if not episode_done else 0.0

        self.Q[s,a] = self.Q[s,a] + self.alpha * (r + self.gamma * Q_next - self.Q[s,a])

        ## update lr for robbins monro conditions
        self.steps += 1
        self.alpha = self.initial_alpha / float(1 + (self.steps // 10000))

        if episode_done:
            ## update epsilon for GLIE
            self.episodes += 1
            self.epsilon = self.initial_epsilon / float(1 + self.episodes)
