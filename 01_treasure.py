import time
import numpy as np
import pandas as pd

np.random.seed(2)

N_STATES = 6  # distance to treasure
ACTIONS = ['left', 'right']  # actions
MAX_EPISODES = 13  # times for learning
FRESH_TIME = 0.3  # fresh time for one move
EPSILON = 0.9  # greedy policy
ALPHA = 0.1  # learning rate
GAMMA = 0.9  # discount factor


# build Q table
def build_q_table(n_states, actions):
    table = pd.DataFrame(
        np.zeros((n_states, len(actions))),  # q_table initial values
        columns=actions  # actions's name
    )
    print(table)
    return table


# how to choose an action
def choose_action(state, q_table):
    state_actions = q_table.iloc[state, :]  # 选择第'state'行的所有列
    # np.random.uniform()生成一个[0, 1)的随机浮点数， state_actions.all()==0说明所有动作的Q值均为0
    if (np.random.uniform() > EPSILON) or (state_actions.all() == 0):
        action_name = np.random.choice(ACTIONS)
    # 返回Q值最大的动作
    else:
        action_name = state_actions.idxmax()
    return action_name
