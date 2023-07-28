from env import Robot
from AgentDQN import DQNAgent
from train import *
from test import *


# Choose what to be done - "train" or "test"
choice = "train"
# File name to be saved/retrieved
fName = '/home/bharath.kumar/code/ZippyRL/DQN_saved_weights'

# training Parameters
episodes = 10000
batch_size = 32
load_file_number_train = 140 # Start your training from --

# testing Parameter
load_file_number_test = 100 # Test your testing from saved_weights from --


if __name__ == "__main__":
    # Create the environment and agent
    env = Robot()
    agent = DQNAgent(env.state_size, env.action_size)

    if choice == "train":
        # Load from where ever its needed

        # Train the agent
        train_agent(env, agent, episodes, batch_size, fileName=fName, verbose=True, load_file_number_train=load_file_number_train)
    elif choice == "test":
        testAgent = test_agent(env, agent, verbose=True, fileName=fName+'/saved_weights'+str(load_file_number))
        testAgent.animate()
    else:
        print(" -%- ")
