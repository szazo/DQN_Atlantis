import os
import tensorflow as tf
import the_agent
import environment
import matplotlib.pyplot as plt
import time
from collections import deque
import numpy as np

GPU_ENABLED = False
if not GPU_ENABLED:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
else:
    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    for device in gpu_devices:
        tf.config.experimental.set_memory_growth(device, True)
    
tf.config.threading.set_inter_op_parallelism_threads(8)

name = 'Atlantis-v0'

experiment = 'deepmind_resized'

agent = the_agent.Agent(experiment,possible_actions=[0,1,2,3],starting_mem_len=100000,max_mem_len=750000,starting_epsilon = 1, learn_rate = .00025)
env = environment.make_env(name,agent)

last_100_avg = [-21]
scores = deque(maxlen = 100)
max_score = -21

# # If testing:
# agent.model.load_weights('recent_weights.hdf5')
# agent.model_target.load_weights('recent_weights.hdf5')
# agent.epsilon = 0.0


env.reset()

for i in range(1000000):
    timesteps = agent.total_timesteps
    timee = time.time()
    score = environment.play_episode(name, experiment, env, agent, debug = False) #set debug to true for rendering
    scores.append(score)
    if score > max_score:
        max_score = score

    print('\nEpisode: ' + str(i))
    print('Steps: ' + str(agent.total_timesteps - timesteps))
    print('Duration: ' + str(time.time() - timee))
    print('Score: ' + str(score))
    print('Max Score: ' + str(max_score))
    print('Epsilon: ' + str(agent.epsilon))

    if i%10==0 and i!=0:
        last_100_avg.append(sum(scores)/len(scores))
        plt.plot(np.arange(0,i+1,10),last_100_avg)
        plt.xlabel('Episode')
        plt.ylabel('Last 100 episode\'s average score')
        plt.savefig('fig_{}{}'.format(experiment, i))
        # plt.show()
