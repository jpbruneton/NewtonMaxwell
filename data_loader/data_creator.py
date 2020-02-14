import numpy as np

# here we create a simple helix data in 3D
set_types = ['E', 'U']
intervals = [[-1, 1], [-2,2]]
steps = [100, 100]

target_x = '2*np.cos(3*t)'
target_y = '3*np.cos(3*t -0.568)'
target_z = '2.3*t'

# first create train function then test functions
for u in range(2):
    if set_types[u] == 'E':
        t = np.linspace(intervals[u][0], intervals[u][1], num=steps[u])
    elif set_types[u] == 'U':
        t = np.random.uniform(intervals[u][0], intervals[u][1], steps[u])
        t = np.sort(t)

    x_t = eval(target_x)
    y_t = eval(target_y)
    z_t = eval(target_z)

    dat = np.transpose(np.array([t, x_t, y_t, z_t]))
    print(dat.shape)
    if u == 0:
        np.savetxt('x1_train(t).csv', dat, delimiter=',')
    else:
        np.savetxt('x1_test(t).csv', dat, delimiter=',')
