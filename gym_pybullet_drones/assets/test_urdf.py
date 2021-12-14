import pybullet as p
import pybullet_data
import time

physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #used by loadURDF
p.setGravity(0,0,0)
planeId = p.loadURDF("ha.urdf")

for i in range (10000):
    p.stepSimulation()
    time.sleep(1./240.)


#p.disconnect()