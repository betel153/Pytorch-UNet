import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

data = np.random.rand(12, 12)

fig, ax = plt.subplots()
heatmap = ax.pcolor(data, cmap=plt.cm.cool)

# ax.set_xticklabels(row_labels, minor=False)
# ax.set_yticklabels(column_labels, minor=False)
# aximg = ax.matshow(data)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
fig.colorbar(heatmap, cax=cax)
plt.show()
plt.savefig('image.png')
