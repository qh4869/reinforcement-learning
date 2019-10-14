import numpy as np
import pandas as pd
import time

np.random.seed(2)

N_STATE = 6
FRESH_TIME = 0.1
N_EPISODE = 10 # 回合
ACTIONS = ['left', 'right']
EPI = 0.9  # probability for choosing actions
LAMBDA = 0.9  # discount factor
ALPHA = 0.1  # learning rate


def update_env(State, episode, step_cnt):
    env_list = ['-']*(N_STATE-1) + ['T']
    if State == 'terminal':
        interaction = 'Episode %s: total_steps = %s' % (episode, step_cnt)
        print('\r{0}'.format(interaction), end='')
        time.sleep(2)
        print('\r                        ', end='')
    else:
        env_list[State] = 'o'
        interaction = ''.join(env_list)
        print('\r{0}'.format(interaction), end='')
        time.sleep(FRESH_TIME)


def build_q_table(actions):
    table = pd.DataFrame(np.zeros((N_STATE-1, len(actions))), columns=actions)
    return table


def choose_action(epi, table, state):
    table_row = table.loc[state, :]
    if np.random.rand() < epi and table_row.any() is True:
        action_res = table_row.idxmax()
    else:
        action_res = np.random.choice(ACTIONS)
    return action_res


def env_forward(action, state):
    if action == 'left':
        reward = 0
        if state != 0:
            next_state = state - 1
        else:
            next_state = state
    else:
        if state != N_STATE-2:
            next_state = state + 1
            reward = 0
        else:
            next_state = 'terminal'
            reward = 1
    return next_state, reward


def main_loop():
    q_table = build_q_table(ACTIONS)
    for episode in range(N_EPISODE):
        is_terminal = False
        state = 0
        step_counter = 0
        # choose the action of the first step
        action = choose_action(EPI, q_table, state)
        while not is_terminal:  # each step
            # forward
            next_state, reward = env_forward(action, state)
            step_counter += 1
            # whether terminal
            if next_state == 'terminal':
                is_terminal = True
            # update q table
            q_now = q_table.loc[state, action]
            if is_terminal:
                q_target = reward
            else:
                # choose the next action
                next_action = choose_action(EPI, q_table, next_state)
                q_target = reward + LAMBDA*q_table.loc[next_state, next_action]
            q_table.loc[state, action] += ALPHA*(q_target-q_now)
            # update state and env
            action = next_action
            state = next_state
            update_env(state, episode, step_counter)
        # show q table of each episode
        print('\r{0}'.format(q_table))
        time.sleep(1)


if __name__ == '__main__':
    main_loop()
