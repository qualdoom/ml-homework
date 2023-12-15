import gym
import numpy as np


def visualize(agent, t_max=10000):
    # для просмотра что моделька делает в целом)
    env_show = gym.make("ALE/Assault-v5", render_mode="human").env
    n_actions = env_show.action_space.n
    
    state = env_show.reset()[0]

    for t in range(t_max):
        env_show.render()
        
        #predict with agent
        
        print(state)
        action = agent.get_action(state, epsilon=0.1)
        print(action)
        
        new_state, reward, done, truncated, _ = env_show.step(action)

        done = done or truncated

        state = new_state
        if done: 
            break
    
    
    env_show.close()