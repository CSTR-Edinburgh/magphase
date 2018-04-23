"""
@author: Felipe Espic
Personal library for plotting
"""
import numpy as np
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


def plot_pitch_marks(v_sig, v_pm_smpls):
    lp.figure()
    lp.plot(v_sig)
    lp.vlines(v_pm_smpls, np.min(v_sig), np.max(v_sig), colors='r')
    lp.grid()
    return
lp.plot_pitch_marks = plot_pitch_marks

