import numpy
import numpy as np
import scipy
from matplotlib.pyplot import plot, show, figure, ylim, xlabel, ylabel
from scipy import signal
from scipy.sparse.csgraph import laplacian
from scipy.fftpack import fftfreq, irfft, rfft
from scipy.stats import pearsonr


def band_pass_filter(signal, fs, fmin, fmax):
    filtered_signals = np.zeros([len(fbands), signal.shape[0]])
    F = fftfreq(signal.shape[0], d=1/fs)
    f_signal = rfft(signal)
    cut_f_signal = f_signal.copy()
    cut_f_signal[(fmin < np.abs(F) < fmax)] = 0
    filtered_signals[i] = irfft(cut_f_signal)
    return filtered_signals


def extract_connectivity(signals, method, fs, fbands):
    n_bands = len(fbands)
    n_channels = signals.shape[1]
    """
    mean = np.matrix(np.mean(signals, axis=1)).T
    signals -= mean @ np.ones([1, signals.shape[1]])
    """

    A = np.zeros([n_bands, n_channels, n_channels])

    if method == 'coh':

        spectrum = np.zeros([n_channels, n_bands])
        for i in range(n_channels):
            ch = signals[:, i]
            f, Pxx = signal.csd(ch, ch, fs)
            for k in range(n_bands):
                Pxx_cut = Pxx.copy()
                Pxx_cut[(np.abs(f) < fbands[k][0]) | (np.abs(f) > fbands[k][1])] = 0
                spectrum[i, k] = np.mean(Pxx_cut)

        for i in range(n_channels):
            A[:, i, i] = 1
            for j in range(i+1, n_channels):
                ch1 = signals[:, i]
                ch2 = signals[:, j]
                f, Pxy = signal.csd(ch1, ch2, fs)
                for k in range(n_bands):
                    Pxy_cut = Pxy.copy()
                    Pxy_cut[(np.abs(f) < fbands[k][0]) | (np.abs(f) > fbands[k][1])] = 0

                    coh = np.abs(np.mean(Pxy_cut))**2/abs(spectrum[i, k] * spectrum[j, k])
                    edge = coh
                    A[k, i, j] = edge
                    A[k, j, i] = edge

    elif method == 'cor':
        for i in range(n_channels):
            A[:, i, i] = 1
            for j in range(i + 1, n_channels):
                ch1 = signals[:, i]
                ch2 = signals[:, j]
                corr, _ = pearsonr(ch1, ch2)
                edge = corr
                for k in range(n_bands):
                    A[k, i, j] = edge
                    A[k, j, i] = edge

    return A





