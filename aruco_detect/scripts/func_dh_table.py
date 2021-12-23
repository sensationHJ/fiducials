import numpy as np

def Rx(theta):
    return np.array([[1, 0, 0 , 0],
                    [0, np.cos(theta),-np.sin(theta), 0],
                    [0, np.sin(theta), np.cos(theta), 0],
                    [0,            0,              0,  1]])
    
def Rz(theta):
    return np.array([[ np.cos(theta), -np.sin(theta), 0 , 0 ],
                    [ np.sin(theta), np.cos(theta) , 0 , 0],
                    [ 0           , 0            , 1 , 0],
                     [0            ,0          ,0     ,1]])

def Tz(z):
    return np.array([[1,0,0,0],
                    [0,1,0,0,],
                    [0,0,1,z],
                    [0,0,0,1]])

def Tx(x):
    return np.array([[1,0,0,x],
                    [0,1,0,0],
                    [0,0,1,0],
                    [0,0,0,1]])

def Amat(theta,d,a,alpha):
    return Rz(theta) @ Tz(d) @ Tx(a) @ Rx(alpha)

def Tmat(dh):
    T = np.eye(4)
    if dh.size < 5:
        T = Amat(dh[0], dh[1], dh[2], dh[3])
    else:
        num_frame, _ = np.shape(dh)
        for k in range(num_frame):
                T = T @ Amat(dh[k,0], dh[k,1], dh[k,2], dh[k,3])

    return T
            
    