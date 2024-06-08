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
        columns=actions  # actions name
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


def get_env_feedback(S, A):
    if A == 'right':
        if S == N_STATES - 2:  # current state
            S_ = 'terminal'
            R = 1  # get the treasure
        else:
            S_ = S + 1
            R = 0
    else:
        R = 0
        if S == 0:  # touch the wall
            S_ = S
        else:
            S_ = S - 1
    return S_, R


# update the environment
def update_env(S, episode, step_counter):
    # update the environment
    env_list = ['-'] * (N_STATES - 1) + ['T']
    if S == 'terminal':
        interaction = 'Episode %s: total_steps = %s' % (episode + 1, step_counter)
        print('\r{}'.format(interaction), end='')
        time.sleep(2)
        print('\r                     ', end='')
    else:
        env_list[S] = 'o'
        interaction = ''.join(env_list)
        print('\r{}'.format(interaction), end='')
        time.sleep(FRESH_TIME)


def rl():
    # main part of loop
    q_table = build_q_table(N_STATES, ACTIONS)
    for episode in range(MAX_EPISODES):
        step_counter = 0
        S = 0                                               # initial state
        is_terminated = False
        update_env(S, episode, step_counter)
        while not is_terminated:
            A = choose_action(S, q_table)
            S_, R = get_env_feedback(S, A)                  # next state, reward
            q_predict = q_table.loc[S, A]
            if S_ != 'terminal':
                q_target = R + GAMMA * q_table.iloc[S_, :].max()    # instant reward + discounted future reward
            else:
                q_target = R
                is_terminated = True

            q_table.loc[S, A] += ALPHA * (q_target - q_predict)     # update Q value
            S = S_

            update_env(S, episode, step_counter + 1)
            step_counter += 1

    return q_table

if __name__ == "__main__":
    q_table = rl()
    print('\r\nQ-table:\n')
    print(q_table)
