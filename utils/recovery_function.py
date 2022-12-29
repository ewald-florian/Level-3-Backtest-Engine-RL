import matplotlib.pyplot as plt
import math
plt.style.use('seaborn')

y_list = []
x_list = []
for x in range(30):
    # Exponential recovery function which is clipped to 0.9.
    y = min(1 - (1/(x+1)) ** 0.6, 0.9)

    x_list.append(x)
    y_list.append(y)

# For the plot.
x_list_1 = [-8, -7, -6, -5, -4, -3, -2, -1, -0.0001]
y_list_1 = [1, 1, 1, 1, 1, 1, 1, 1, 1]

x = x_list_1 + x_list
y = y_list_1 + y_list
plt.plot(x, y)
plt.ylabel("Recovery Factor")
plt.xlabel("Seconds before/after execution")
plt.title("Clipped Exponential Recovery Function", fontsize=14)
plt.show()


