FROM nvcr.io/nvidia/pytorch:19.10-py3

RUN pip install --upgrade pip
RUN pip install --upgrade numpy
RUN pip install pybullet
RUN pip install stable-baselines3[extra]

RUN pip install matplotlib
RUN pip install seaborn