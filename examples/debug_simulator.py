import time
#import gym
import numpy as np
import argparse
import pybullet as p
from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics
from gym_pybullet_drones.envs.single_agent_rl.StraightFlightAviary import StraightFlightAviary
#from gym_pybullet_drones.envs.single_agent_rl.TakeoffAviary import TakeoffAviary
from gym_pybullet_drones.utils.utils import sync
from gym.envs.registration import register
from gym_pybullet_drones.utils.utils import sync, str2bool

if __name__ == "__main__":


    parser = argparse.ArgumentParser(description='Helix flight script using CtrlAviary or VisionAviary and DSLPIDControl')
    parser.add_argument('--drone',              default="ha",       type=DroneModel,    help='Drone model (default: CF2X)', metavar='', choices=DroneModel)
    parser.add_argument('--physics',            default="pyb",      type=Physics,       help='Physics updates (default: PYB)', metavar='', choices=Physics)
    parser.add_argument('--gui',                default=True,       type=str2bool,      help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--aggregate',          default=True,       type=str2bool,      help='Whether to aggregate physics steps (default: True)', metavar='')
    parser.add_argument('--simulation_freq_hz', default=240,        type=int,           help='Simulation frequency in Hz (default: 240)', metavar='')
    parser.add_argument('--control_freq_hz',    default=48,         type=int,           help='Control frequency in Hz (default: 48)', metavar='')
    parser.add_argument('--duration_sec',       default=12,         type=int,           help='Duration of the simulation in seconds (default: 5)', metavar='')
    parser.add_argument('--trajectory',         default=1,          type=int,           help='Trajectory type (default: 1)', metavar='')
    parser.add_argument('--wind',               default=False,      type=str2bool,      help='Whether to enable wind (default: False)', metavar='')
    parser.add_argument('--record_video',       default=False,      type=str2bool,      help='Whether to record a video (default: False)', metavar='')

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
    nondeterministic = False)

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

    CTRL_EVERY_N_STEPS = int(np.floor(env.SIM_FREQ/ARGS.control_freq_hz))
    print("[INFO] Action space:", env.action_space)
    print("[INFO] Observation space:", env.observation_space)

    PYB_CLIENT = env.getPyBulletClient()
    p.setGravity(0,0,0)
    #obs = env.reset()
    action = np.array([0,0])
    start = time.time()
    for i in range(0, int(ARGS.duration_sec*env.SIM_FREQ), AGGR_PHY_STEPS):


        obs, reward, done, info = env.step(action)
        if i%CTRL_EVERY_N_STEPS == 0:
            action = np.array([10,10])

        
        

        if i%env.SIM_FREQ == 0:
            env.render()
            print(done)
        sync(i, start, env.TIMESTEP)
        if done:
            obs = env.reset()
    env.close()


    
