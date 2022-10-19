from setuptools import setup

setup(name='gym_pybullet_drones',
    version='1.0.0',
    install_requires=[
        'numpy',
        'Pillow',
        'matplotlib',
        'cycler',
        'gym==0.21.0',
        'pybullet',
        'stable_baselines3==1.6.0',
        'ray[rllib]'
        ]
)
