"""Script demonstrating the use of `gym_pybullet_drones`' Gym interface.

Class TakeoffAviary is used as a learning env for the A2C and PPO algorithms.

Example
-------
In a terminal, run as:

    $ python learn.py

Notes
-----
The boolean argument --rllib switches between `stable-baselines3` and `ray[rllib]`.
This is a minimal working example integrating `gym-pybullet-drones` with 
reinforcement learning libraries `stable-baselines3` and `ray[rllib]`.
It is not meant as a good/effective learning example.

"""
import time
import gym
import numpy as np
from stable_baselines3 import A2C, PPO, DDPG, SAC
#from stable_baselines3.ddpg import MlpPolicy
from stable_baselines3.a2c import MlpPolicy
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import VecCheckNan, DummyVecEnv
from gym_pybullet_drones.envs.single_agent_rl.StraightFlightAviary import StraightFlightAviary

from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.envs.single_agent_rl.TakeoffAviary import TakeoffAviary
from gym_pybullet_drones.utils.utils import sync
from gym.envs.registration import register

if __name__ == "__main__":

    register(
    id='straight-flight-aviary-v0',
    entry_point='gym_pybullet_drones.envs.single_agent_rl:StraightFlightAviary',
    reward_threshold=1.0,
    nondeterministic = False,
)

    #### Check the environment's spaces ########################
    env = gym.make("straight-flight-aviary-v0")

    print("[INFO] Action space:", env.action_space)
    print("[INFO] Observation space:", env.observation_space)
    check_env(env,
              warn=True,
              skip_render_check=True
              )

    #### Train the model #######################################
    model = DDPG("MlpPolicy",
                    env,
                    verbose=1,
                    tensorboard_log="./a2c_StraightFlight_tensorboard/"
                    )
    model.learn(total_timesteps=1000000) # Typically not enough

    #### Show (and record a video of) the model's performance ##
    env = StraightFlightAviary(gui=True,
                        record=False
                        )
    logger = Logger(logging_freq_hz=int(env.SIM_FREQ/env.AGGR_PHY_STEPS),
                    num_drones=1
                    )
    obs = env.reset()
    start = time.time()
    for i in range(20*env.SIM_FREQ):
        action, _states = model.predict(obs,
                                            deterministic=True
                                            )

        obs, reward, done, info = env.step(action)
        logger.log(drone=0,
                   timestamp=i/env.SIM_FREQ,
                   state=np.hstack([obs[0:3], np.zeros(4), obs[3:15],  np.resize(action, (2))]),
                   control=np.zeros(12)
                   )
        if i%env.SIM_FREQ == 0:
            env.render()
            print(done)
        sync(i, start, env.TIMESTEP)
        if done:
            obs = env.reset()
    env.close()
    logger.plot()
