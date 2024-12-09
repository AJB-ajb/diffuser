import numpy as np
from gridworld import GridWorld

world=\
    """
    wwwwwwwwwwwwwwwww
    wa   o   w     gw
    w               w
    www  o   www  www
    w               w
    wwwww    o    www
    w     ww        w
    wwwwwwwwwwwwwwwww
    """
    
env=GridWorld(world,slip=0.2) # Slip is the degree of stochasticity of the gridworld.

# Value Iteratioin
V=np.zeros((env.state_count,1))
V_prev=np.random.random((env.state_count,1))
eps=1e-7
gamma=0.9

while np.abs(V-V_prev).sum()>eps:
    Q_sa=env.R_sa+gamma*np.squeeze(np.matmul(env.P_sas,V),axis=2)
    V_prev=V.copy()
    V=np.max(Q_sa,axis=1,keepdims=True)
    pi=np.argmax(Q_sa,axis=1)

print("Pi:",pi)
env.show(pi)