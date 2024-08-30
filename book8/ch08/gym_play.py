'''
modified the original code to use gymnasium instead of OpenAI gym(deprecated).

original code:https://github.com/oreilly-japan/deep-learning-from-scratch-4/blob/master/ch08/gym_play.py
'''
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import time
import matplotlib
# matplotlib.use("Qt5agg")
env = gym.make('CartPole-v1', render_mode = 'rgb_array')



fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_axis_off()
plt.show(block=False)
while True:
    state = env.reset()
    terminated = False
    truncated = False
    while not terminated or truncated:
        rend = env.render()
        action = env.action_space.sample()
        state, reward, terminated, truncated, info = env.step(action)
        print()
        print('state:', state)
        print('reward:', reward)
        print('terminated:', terminated)
        print('truncated:', truncated)
        print('info:', info)
        print()
        
        image =  rend
        ax.imshow(image,cmap='gray',)
        plt.gcf().canvas.draw_idle()
        plt.gcf().canvas.start_event_loop(0.2)
