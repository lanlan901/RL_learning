import numpy as np
import pandas as pd


class RL(object):
    def __init__(self, action_space, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = action_space
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def choose_action(self, observation):
        self.check_state_exist(observation)
        if np.random.rand() < self.epsilon:  # Q max action
            state_action = self.q_table.loc[observation, :]
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)
        else:  # random exploration
            action = np.random.choice(self.actions)
        return action

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            self.q_table = self.q_table._append(
                pd.Series([0] * len(self.actions), index=self.q_table.columns, name=state))

    def learn(self, *args):
        pass

# on-policy
class SarsaLambdaTable(RL):
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9, trace_dacay=0.9):
        super(SarsaLambdaTable, self).__init__(actions, learning_rate, reward_decay, e_greedy)

        # backward view, eligibility trace.
        self.lambda_ = trace_dacay
        self.eligibility_trace = self.q_table.copy()

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            to_be_append = pd.Series([0] * len(self.actions), index=self.q_table.columns, name=state)
            self.q_table = self.q_table._append(to_be_append)
            self.eligibility_trace = self.eligibility_trace._append(to_be_append)

    def learn(self, s, a, r, s_, a_):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        if s_ != "terminal":
            q_target = r + self.gamma * self.q_table.loc[s_, a_]
        else:
            q_target = r

        error = q_target - q_predict
        # 将当前状态的所有动作的资格迹值重置为 0。
        # 将当前状态下执行的动作的资格迹值设置为 1。
        self.eligibility_trace.loc[s, :] *= 0
        self.eligibility_trace.loc[s, a] = 1

        self.q_table += self.lr * error * self.eligibility_trace
        self.eligibility_trace *= self.gamma * self.lambda_
