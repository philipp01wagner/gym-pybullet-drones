FROM nvcr.io/nvidia/pytorch:22.06-py3

#ENV NVIDIA_VISIBLE_DEVICES=$GPU4,$GPU5,$GPU6MIG0,$GPU6MIG1,$GPU6MIG2,$GPU7MIG0,$GPU7MIG1,$GPU7MIG2


#RUN git clone https://github.com/philipp01wagner/gym-pybullet-drones.git
COPY . .

RUN apt update

RUN pip3 install -U pip

RUN pip3 install --upgrade numpy Pillow matplotlib cycler 
RUN pip3 install --upgrade gym==0.21.0 pybullet stable_baselines3 'ray[rllib]'
ENV DEBIAN_FRONTEND=noninteractive
RUN apt install -y ffmpeg
RUN apt install -y xdg-utils


RUN pip3 install -e .