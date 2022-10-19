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
import sys, os
sys.path.append(os.getcwd())
import time
import datetime
import gym
import numpy as np
import argparse
from stable_baselines3 import A2C, PPO, DDPG, SAC, TD3
from stable_baselines3.common.env_checker import check_env
from gym_pybullet_drones.envs.single_agent_rl.StraightFlightAviary import StraightFlightAviary

from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics
#from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync
from gym.envs.registration import register
from gym_pybullet_drones.utils.utils import sync, str2bool
from wandb.integration.sb3 import WandbCallback


def find_alg_name(alg):
    s = str(alg)
    ind1 = s[::-1].find(".")
    name = s[-ind1:-2]
    return name

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Helix flight script using CtrlAviary or VisionAviary and DSLPIDControl')
    parser.add_argument('--drone',              default="ha",       type=DroneModel,    help='Drone model (default: CF2X)', metavar='', choices=DroneModel)
    parser.add_argument('--physics',            default="pyb",      type=Physics,       help='Physics updates (default: PYB)', metavar='', choices=Physics)
    parser.add_argument('--gui',                default=False,       type=str2bool,     help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--aggregate',          default=True,       type=str2bool,      help='Whether to aggregate physics steps (default: True)', metavar='')
    parser.add_argument('--simulation_freq_hz', default=240,        type=int,           help='Simulation frequency in Hz (default: 240)', metavar='')
    parser.add_argument('--control_freq_hz',    default=24,         type=int,           help='Control frequency in Hz (default: 48)', metavar='')
    parser.add_argument('--duration_sec',       default=48,         type=int,           help='Duration of the simulation in seconds (default: 5)', metavar='')
    parser.add_argument('--trajectory',         default=1,          type=int,           help='Trajectory type (default: 1)', metavar='')
    parser.add_argument('--wind',               default=False,      type=str2bool,      help='Whether to enable wind (default: False)', metavar='')
    parser.add_argument('--record_video',       default=False,      type=str2bool,      help='Whether to record a video (default: False)', metavar='')
    parser.add_argument('--alg',                default="a2c",      type=str,           help='Which algorithm to use (default: A2C)', metavar='')
    parser.add_argument('--timesteps',          default=1000,      type=int,           help='Number of training timesteps (default: 10.000)', metavar='')

    ARGS = parser.parse_args()

    H = 1.0
    R = .3
    INIT_XYZS = np.array([[0, 0, H]])
    INIT_RPYS = np.array([[0, 0,  0]])
    AGGR_PHY_STEPS = int(ARGS.simulation_freq_hz/ARGS.control_freq_hz) if ARGS.aggregate else 1

    register(
    id='straight-flight-aviary-v0',
    entry_point='gym_pybullet_drones.envs.single_agent_rl:StraightFlightAviary',
    reward_threshold=1.0,
    nondeterministic = False,
)

    #### Check the environment's spaces ########################
    env = StraightFlightAviary(drone_model=ARGS.drone,
                         initial_xyzs=INIT_XYZS,
                         initial_rpys=INIT_RPYS,
                         physics=ARGS.physics,
                         freq=ARGS.simulation_freq_hz,
                         aggregate_phy_steps=AGGR_PHY_STEPS,
                         gui=ARGS.gui,
                         record=ARGS.record_video
                         )

    print("[INFO] Action space:", env.action_space)
    print("[INFO] Observation space:", env.observation_space)
    check_env(env,
              warn=True,
              skip_render_check=True
              )

    algs = [A2C, PPO, SAC, DDPG, TD3]

    algorithm = [a for a in algs if find_alg_name(a).lower() in ARGS.alg][0]
    #### Train the model #######################################
    model = algorithm("MlpPolicy",
                    env,
                    verbose=1,
                    tensorboard_log="./a2c_StraightFlight_tensorboard/"
                    )
    model.learn(total_timesteps=ARGS.timesteps, callback=WandbCallback()) # Typically not enough

    #### Show (and record a video of) the model's performance ##
    env = StraightFlightAviary(gui=ARGS.gui,
                        record=ARGS.record_video
                        )
    #logger = Logger(logging_freq_hz=int(env.SIM_FREQ/env.AGGR_PHY_STEPS),
    #                num_drones=1
    #                )
    obs = env.reset()
    start = time.time()
    for i in range(20*env.SIM_FREQ):
        action, _states = model.predict(obs,
                                            deterministic=True
                                            )

        obs, reward, done, info = env.step(action)
        #logger.log(drone=0,
        #           timestamp=i/env.SIM_FREQ,
        #           state=np.hstack([obs[0:3], np.zeros(4), obs[3:15],  np.resize(action, (2))]),
        #           control=np.zeros(12)
        #           )
        if i%env.SIM_FREQ == 0:
            env.render()

        sync(i, start, env.TIMESTEP)
        if done:
            obs = env.reset()
    env.close()
    #logger.plot()

    
    timestring = datetime.datetime.now().strftime('%m_%d_%Y_%H_%M_%S') 
    model_name = find_alg_name(algorithm) + "_" + timestring
    model.save(model_name)

