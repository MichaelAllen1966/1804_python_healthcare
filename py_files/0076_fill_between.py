import matplotlib.pyplot as plt

x = [0, 10, 20, 40, 50]

y1 = [60, 65, 65, 70, 75]
y1_max = [70, 73, 70, 78, 82]
y1_min = [55, 58, 61, 62, 68]

y2 = [45, 50, 55, 55, 55]
y2_max = [53, 55, 63, 60, 62]
y2_min = [35, 45, 52, 52, 50]

plt.plot(x, y1)
plt.plot(x, y2, linestyle ='dashed')

# alpha adjusts transparency, higher alpha --> darker grey
# Or color could be set to, for example '0.2', but using transparency allows
# overlapping shaded areas
plt.fill_between(x, y1_min, y1_max, color = 'k', alpha = 0.1)
plt.fill_between(x, y2_min, y2_max, color = 'k', alpha = 0.1)

plt.savefig('plot_29.png')
plt.show()