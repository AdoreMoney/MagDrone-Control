This repository contains the implementation of a hybrid control strategy,  
combining PID and reinforcement learning for elevator-only underactuated tandem-wing UAVs in oceanic magnetic survey missions(called MagDrone).  
The control system leverages elevator differential-synchronous actuation for regulation. 
This project includes  
--dynamics modeling,  
--trim calculations,  
--small perturbation linearization,  
--lateral and longitudinal PID control design,  
--optimization of a deep reinforcement learning PPO algorithm  
for the MagDrone.   
NOTE:  
    To protect intellectual property rights, the MagDrone's aerodynamic parameters are replaced with 0.0 here.  
    This repository is solely for demonstrating the experimental structure.  
    In practice, the parameters can be modified to suit specific drones.  
