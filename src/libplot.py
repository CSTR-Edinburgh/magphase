"""
@author: Felipe Espic
Personal library for plotting
"""
import matplotlib
matplotlib.use("Qt4Agg")
from matplotlib import pyplot as lp
from mpl_toolkits.mplot3d import Axes3D
lp.ion()

matplotlib.rcParams['lines.antialiased'] = False
matplotlib.rcParams['lines.linewidth']   = 1.0

# FOR GENERAL USE:====================================================
def plotm(m_data):
    lp.figure()
    ret = lp.imshow(m_data.T, cmap=lp.cm.inferno, aspect='auto', origin='lower', interpolation='nearest')
    lp.colorbar(ret)
    return
lp.plotm = plotm
