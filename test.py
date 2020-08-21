import matplotlib.pyplot as plt

x = [30, 36, 40, 44, 50, 58, 60, 61, 63, 65, 66, 68, 72, 75, 75, 78, 70, 80, 84, 90, 93,
      32, 39, 41, 42, 51, 53, 61, 64, 66, 67, 68, 69, 73, 74, 78, 79, 80, 81, 90, 22, 84,]

fig = plt.figure()
ax = fig.add_subplot(1,1,1)

ax.hist(x, bins=10)
ax.set_title('test histogram')
ax.set_xlabel('Score')
ax.set_ylabel('Num of Poeple')
fig.show()
