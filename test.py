import numpy as np
import matplotlib.pyplot as plt

def f(x, y, mu, S):
  x_norm = (np.array([x, y]) - mu[:, None, None]).transpose(1, 2, 0)
  return np.exp(- x_norm[:, :, None, :] @ np.linalg.inv(S)[None, None, :, :] @ x_norm[:, :, :, None] / 2.0) / (2*np.pi*np.sqrt(np.linalg.det(S)))

x = y = np.arange(-20, 20, 0.5)
X, Y = np.meshgrid(x, y)

mu = np.array([0,0])
S = np.array([[20,0],[0,20]])

Z = f(X,Y, mu, S)[:, :, 0, 0]

fig1 = plt.figure()
ax1 = plt.subplot(111, projection='3d')
ax1.plot_surface(X,Y,Z, antialiased=False, cmap="plasma")

fig2 = plt.figure()
ax2 = plt.subplot(111)
cs = ax2.contourf(X,Y,Z,100, cmap="plasma")
plt.colorbar(cs)

plt.show()
