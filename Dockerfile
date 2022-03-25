FROM nvcr.io/nvidia/pytorch:19.10-py3

RUN pip3 install --upgrade pip
RUN pip3 install --upgrade numpy
RUN pip3 install pybullet
RUN pip3 install stable-baselines3[extra]

RUN pip3 install matplotlib
RUN pip3 install seaborn