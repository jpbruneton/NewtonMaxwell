import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def f(y, t):
    x1, y1, z1, x2, y2, z2, vx1, vy1, vz1, vx2, vy2, vz2 = y      # unpack current values of y
    dist = np.sqrt((x1-x2)**2 +(y1-y2)**2 +(z1-z2)**2)
    G = 6.67*1e-11
    m2 = 1e30
    m1 = 2e30
    derivs = [vx1, vy1, vz1,  vx2, vy2, vz2, - G*m2*(x2 - x1)/dist**3, - G*m2*(y2 - y1)/dist**3, - G*m2*(z2 - z1)/dist**3,
              G*m1*(x2 - x1)/dist**3, G*m1*(y2 - y1)/dist**3, - G*m1*(z2 - z1)/dist**3]    # list of dy/dt=f functions
    return derivs


# Initial values
theta0 = 0.0     # initial angular displacement
omega0 = 0.0     # initial angular velocity

# Bundle initial conditions for ODE solver
y0 = [0, 1e4, 0, 0, -1e4, 0, 1e15, 0, 0, 0, 0, 0]


# Make time array for solution
tStop = 2000.
tInc = 1
t = np.arange(0., tStop, tInc)

# Call the ODE solver
psoln = odeint(f, y0, t, args=())
print(psoln.shape)

# Plot results
fig = plt.figure(1, figsize=(8,8))

# Plot theta as a function of time
ax1 = fig.add_subplot(311)
ax1.plot(t, psoln[:,1])
ax1.set_xlabel('time')
ax1.set_ylabel('theta')

# Plot omega as a function of time
ax2 = fig.add_subplot(312)
ax2.plot(t, psoln[:,1])
ax2.set_xlabel('time')
ax2.set_ylabel('omega')

# Plot omega vs theta
ax3 = fig.add_subplot(313)
twopi = 2.0*np.pi
ax3.plot(psoln[:,0]%twopi, psoln[:,1], '.', ms=1)
ax3.set_xlabel('theta')
ax3.set_ylabel('omega')
ax3.set_xlim(0., twopi)

plt.tight_layout()
plt.show()