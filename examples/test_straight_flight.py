import time
import gym
import numpy as np
import argparse
from stable_baselines3 import A2C, PPO, DDPG, SAC, TD3
from stable_baselines3.common.env_checker import check_env
import pybullet as p
from gym_pybullet_drones.envs.single_agent_rl.StraightFlightAviary import StraightFlightAviary

from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics
from gym_pybullet_drones.utils.utils import sync
from gym.envs.registration import register
from gym_pybullet_drones.utils.utils import sync, str2bool#


def find_alg_name(alg):
    s = str(alg)
    ind1 = s[::-1].find(".")
    name = s[-ind1:-2]
    return name

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Helix flight script using CtrlAviary or VisionAviary and DSLPIDControl')
    parser.add_argument('--drone',              default="ha",       type=DroneModel,    help='Drone model (default: CF2X)', metavar='', choices=DroneModel)
    parser.add_argument('--physics',            default="pyb",      type=Physics,       help='Physics updates (default: PYB)', metavar='', choices=Physics)
    parser.add_argument('--gui',                default=False,       type=str2bool,      help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--aggregate',          default=True,       type=str2bool,      help='Whether to aggregate physics steps (default: True)', metavar='')
    parser.add_argument('--simulation_freq_hz', default=240,        type=int,           help='Simulation frequency in Hz (default: 240)', metavar='')
    parser.add_argument('--control_freq_hz',    default=48,         type=int,           help='Control frequency in Hz (default: 48)', metavar='')
    parser.add_argument('--duration_sec',       default=200,         type=int,           help='Duration of the simulation in seconds (default: 5)', metavar='')
    parser.add_argument('--trajectory',         default=1,          type=int,           help='Trajectory type (default: 1)', metavar='')
    parser.add_argument('--wind',               default=False,      type=str2bool,      help='Whether to enable wind (default: False)', metavar='')
    parser.add_argument('--record_video',       default=False,      type=str2bool,      help='Whether to record a video (default: False)', metavar='')
    parser.add_argument('--policy_path',        default="",         type=str,           help='path to the policy zip file', metavar='')

    ARGS = parser.parse_args()
    policy_name = ARGS.policy_path

    algs = [A2C, PPO, SAC, DDPG, TD3]


    algorithm = [a for a in algs if find_alg_name(a).lower() in policy_name][0]


    H = 1.0
    R = .3
    INIT_XYZS = np.array([[0, 0, H]])
    INIT_RPYS = np.array([[0, 0,  0]])
    AGGR_PHY_STEPS = int(ARGS.simulation_freq_hz/ARGS.control_freq_hz) if ARGS.aggregate else 1

    env = StraightFlightAviary(gui=True,
                        record=False,
                        initial_xyzs=INIT_XYZS,
                        initial_rpys=INIT_RPYS,
                        physics=ARGS.physics,
                        freq=ARGS.simulation_freq_hz,
                        aggregate_phy_steps=AGGR_PHY_STEPS,
                        duration_sec=ARGS.duration_sec
                        )

    PYB_CLIENT = env.getPyBulletClient()
    p.setGravity(0,0,0)

    model = algorithm.load(policy_name)

    #obs = env.reset()
    action = np.array([0,0])
    start = time.time()
    for i in range(200*env.SIM_FREQ):
        obs, reward, done, info = env.step(action)

        if i%env.SIM_FREQ == 0:
            env.render()
            print(done)
        sync(i, start, env.TIMESTEP)
        if done:
            print("DONE")
            obs = env.reset()
        action, _states = model.predict(obs,
                                            deterministic=True
                                            )

#        print("X: ", obs[0])
#        print("Reward: ", reward)
        
    env.close()
