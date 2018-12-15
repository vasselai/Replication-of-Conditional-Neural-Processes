"""
Plotting helper functions
"""
import matplotlib.pyplot as plt


def reg_plotting(it, Xt, Yt, Xc, Yc, Yhat, var):
  plt.plot(Xt[0], Yhat[0], 'b', linewidth=2)
  plt.plot(Xc[0], Yc[0], 'ko', markersize=10)
  plt.plot(Xt[0], Yt[0], 'k', linewidth=2)
  plt.fill_between(Xt[0, :, 0], Yhat[0, :, 0] - var[0, :, 0], \
                   Yhat[0, :, 0] + var[0, :, 0], alpha=0.2, \
                   facecolor='#b7b7f7', interpolate=True)
  plt.grid(False)
  plt.savefig('D:/Michigan/Classes/5th Semester/EECS 545/Final Project/code/real/figures/plot_' + str(it) + '.png')
  plt.gcf().clear()