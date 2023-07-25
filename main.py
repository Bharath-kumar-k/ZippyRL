from env import Robot
from AgentDQN import DQNAgent
from train import *


# Create the environment and agent
env = Robot()
agent = DQNAgent(env.state_size, env.action_size)

# Train the agent
episodes = 10000
batch_size = 32
fName = '/home/bharath.kumar/code/ZippyRL/DQN_saved_weights'
# train_agent(env, agent, episodes, batch_size, fileName=fName, verbose=True)
test_agent(env, agent, verbose=True, fileName=fName+'/saved_weights'+str(600), plotFlag=True)

# # Define the parameters
# target = 10
# v0 = 0.0
# a0 = 0.0

# Save the model
# test_agent(env, agent)
# Load the saved model
