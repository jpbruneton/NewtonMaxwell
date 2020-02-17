import numpy as np
import matplotlib.pyplot as plt
# here we create a simple helix data in 3D
set_types = ['E', 'U']
intervals = [[-1, 1], [-2,2]]
steps = [1000, 1000]

from scipy.optimize import root

def keplerian_motion(eccentricty, v):
    # in cartesian cord : from page 16-19 of https://arxiv.org/pdf/1609.00915.pdf
    def ff(x,t):
        return x - eccentricty*np.sin(x) - v *t

    t = np.linspace(0, 7, num=1000)
    sols=[]
    for elem in t:
        def f(x):
            return ff(x,elem)

        sol = root(f, 0)
        sols.append(sol.x)
    theta = []
    for elem in sols:
        theta.append(2*np.arctan(np.tan(elem/2)*np.sqrt((1 + eccentricty)/(1-eccentricty))))

    radius = 10/(1+ eccentricty*np.cos(theta))
    #plt.polar(theta, radius)
    #plt.show()
    x = list(radius*np.cos(theta))
    y = list(radius*np.sin(theta))
    z= [0]*1000
    t = list(t)
    data = np.transpose(np.array([t, x, y,z]))
    np.savetxt('kepler_1.csv', data, delimiter=',')


keplerian_motion(0.7,1)
target_x = '5*np.cos(2*t)'
target_y = '1.2*np.cos(2*t -0.568)'
target_z = '0'

# first create train function then test functions
for u in range(2):
    if set_types[u] == 'E':
        t = np.linspace(intervals[u][0], intervals[u][1], num=steps[u])
    elif set_types[u] == 'U':
        t = np.random.uniform(intervals[u][0], intervals[u][1], steps[u])
        t = np.sort(t)

    x_t = eval(target_x)
    plt.plot(x_t)
    plt.show()
    y_t = eval(target_y)
    z_t = eval(target_z)

    dat = np.transpose(np.array([t, x_t, y_t, z_t]))
    print(dat.shape)
    if u == 0:
        np.savetxt('x1_train(t).csv', dat, delimiter=',')
    else:
        np.savetxt('x1_test(t).csv', dat, delimiter=',')
