import numpy as np
import matplotlib.pyplot as plt

import logging

formatter = logging.Formatter('%(asctime)s:%(name)-8s:%(message)-15s')

train_file_handler = logging.FileHandler('training.log', mode='w')
train_file_handler.setFormatter(formatter)
train_logger = logging.getLogger(__name__)
train_logger.setLevel(logging.DEBUG)
train_logger.addHandler(train_file_handler)

def train_agent(env, agent, episodes, batch_size, verbose=False, fileName=''):
    # Logging parameters
    log_frequency = 10
    episode_rewards = []
    avg_rewards = []

    for episode in range(episodes):
        state = env.reset()
        state = np.reshape(state, [1, env.state_size])
        done = False
        episode_reward = 0
        t = 0  # Time step counter
        rewardBreakups = dict(rGoalDistance=0,
                              rSlowerThanAvgTime=0,
                              rVLimitCrossing=0,
                              rALimitCrossing=0,
                              rUnnesAcc=0,
                              rMovingBack=0,
                              rMovingBeyondGoal=0,
                              rReachingGoal=0,
                              rReachingGoalWithAcc=0,
                              rReachingGoalWithVel=0)
        while not done:
            # Choose an action
            action = agent.act(state)
            # Perform the action in the environment
            next_state, time, reward, done, rBreakups = env.step(action)
            next_state = np.reshape(next_state, [1, env.state_size])
            # Store the experience in the agent's memory
            agent.remember(state, action, reward, next_state, done)

            state = next_state
            t += 1

            episode_reward += reward
            for key in rewardBreakups:
                rewardBreakups[key] += rBreakups[key]

            if done:
                if verbose:
                    print(rewardBreakups)
                    print("Time: {}, Distance to goal: {}, Velocity: {}, Acceleration: {}, Action: {}".format(
                        np.round(time, 2), np.round(state[0][0], 2), np.round(state[0][1], 2), np.round(state[0][2], 2),
                        action))
                # print("Episode {}/{} completed | Episode Reward: {}".format(episode + 1, episodes, episode_reward))
                train_logger.info(f'Episode {episode + 1}/{episodes} completed | Episode Reward: {episode_reward}')

                episode_rewards.append(episode_reward)
                break

        if verbose:
            print("[--------------------------------------------------------------------------------------------------]")
        if len(agent.memory) > batch_size:
            # Train the agent by replaying experiences from memory
            agent.replay(batch_size)

        # Logging and visualization
        if (episode + 1) % log_frequency == 0:
            avg_reward = np.mean(episode_rewards[-log_frequency:])
            avg_rewards.append(avg_reward)
            # print("Average Reward (Episodes {}-{}): {}".format(episode - log_frequency + 2, episode + 1, avg_reward))
            agent.save(name=fileName+'/saved_weights'+str(episode + 1))
            train_logger.info(f'Average Reward (Episodes {episode - log_frequency + 2}-{episode + 1}): {avg_reward}')
            print("--------------------------------------------- Model Saved --------------------------------------------- ")

        # # Logging and visualization
        # if (episode + 1) % log_frequency == 0:
        #     avg_reward = np.mean(episode_rewards[-log_frequency:])
        #     avg_rewards.append(avg_reward)
        #     print("Average Reward (Episodes {}-{}): {}".format(episode - log_frequency + 2, episode + 1, avg_reward))


def test_agent(env, agent, verbose=True, fileName='', plotFlag=True):
    state = env.reset()
    state = np.reshape(state, [1, env.state_size])
    agent.load(fileName)
    done = False
    t = 0  # Time step counter
    timeVec = []
    distToGoalVec = []
    velVec = []
    accVec = []
    actionVec =[]
    while not done:
        # Choose an action
        action = agent.actual_act(state)
        # Perform the action in the environment
        next_state, time, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, [1, env.state_size])

        state = next_state
        t += 1

        timeVec.append(np.round(time, 2))
        distToGoalVec.append(np.round(state[0][0], 2))
        velVec.append(np.round(state[0][1], 2))
        accVec.append(np.round(state[0][2], 1))
        actionVec.append(action)

        if verbose:
            # print("Time: {}, Distance to goal: {}, Velocity: {}, Acceleration: {}, Action: {}".format(
            #     np.round(time, 2), np.round(state[0][0], 2), np.round(state[0][1], 2), np.round(state[0][2], 2),
            #     action))
            print("Time: {}, Distance to goal: {}, Velocity: {}, Acceleration: {}, Action: {}".format(
                timeVec[-1], distToGoalVec[-1], velVec[-1], accVec[-1],
                action))
        if done:
            break

    if plotFlag:
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1)
        ax1.plot(timeVec, distToGoalVec)
        ax2.plot(timeVec, velVec)
        ax3.plot(timeVec, accVec)
        ax4.plot(timeVec, actionVec)

        ax1.grid(True)
        ax2.grid(True)
        ax3.grid(True)
        ax4.grid(True)

        plt.show()
