import numpy as np
import numba as nb
from scipy.interpolate import CubicSpline

@nb.njit(parallel=True)
def make_complex(cmpl, mag, phase):
    N = cmpl.shape[0]
    for i in nb.prange(0, N):
        cmpl[i] = mag[i] * np.exp(1j * phase[i])


def get_coarse_xcorr(f1, f2, chans):
    """Get coarse xcorr of each channel of two channelized timestreams.
    The xcorr is 0-padded, so length of output is twice the original length (shape[0]).

    Parameters
    ----------
    f1, f2 : ndarray of complex64
        First and second timestreams. Both n_spectrum x n_channel complex array.
    chans: tuple of int
        Channels (columns) of f1 and f2 that should be correlated.

    Returns
    -------
    ndarray of complex128
        xcorr of each channel's timestream. 2*n_spectrum x n_channel complex array.
    """
    Nsmall = f1.shape[0]
    #print("Shape of passed channelized timestream =", f1.shape)
    xcorr = np.zeros((len(chans),2 * Nsmall), dtype="complex128")
    wt = np.zeros(2 * Nsmall)
    wt[:Nsmall] = 1
    n_avg = np.fft.irfft(np.fft.rfft(wt) * np.conj(np.fft.rfft(wt)))
    #print("n_avg is", n_avg)
    for i, chan in enumerate(chans):
        #print("processing chan", chan)
        xcorr[i, :] = np.fft.ifft(
            np.fft.fft(
                np.hstack([f1[:, chan].flatten(), np.zeros(Nsmall, dtype="complex128")])
            )
            * np.conj(
                np.fft.fft(
                    np.hstack(
                        [f2[:, chan].flatten(), np.zeros(Nsmall, dtype="complex128")]
                    )
                )
            )
        )
        xcorr[i, :] = xcorr[i, :] * 2 * Nsmall / n_avg # get rid of the N introduced by ifft first then divide by correct N
    return xcorr


def get_interp_xcorr(coarse_xcorr, chan, sample_no, coarse_sample_no):
    """Get a upsampled xcorr from coarse_xcorr by adding back the carrier frequency.

    Parameters
    ----------
    coarse_xcorr: ndarray
        1-D array of coarse xcorr of one channel.
    chan : int
        Channel for the passed coarse xcorr.
    osamp: int
        Number of times to over sample over the default 4 ns time-resolution. E.g. osamp=4 means 1 ns time-resolution.

    Returns
    -------
    final_xcorr_cwave: ndarray
        Complex upsampled xcorr.
    """
    print("coarse shape", coarse_xcorr.shape)
    final_xcorr_cwave = np.empty(
        sample_no.shape[0], dtype="complex128"
    )
    print("Total upsampled timestream samples in this coarse chunk =", sample_no.shape)
    uph = np.unwrap(np.angle(coarse_xcorr))  # uph = unwrapped phase
    newphase = 2 * np.pi * chan * np.arange(0, coarse_xcorr.shape[0]) + uph
    newphase = np.interp(sample_no, coarse_sample_no, newphase)
    cs = CubicSpline(coarse_sample_no, np.abs(coarse_xcorr))
    newmag = cs(sample_no)
    make_complex(final_xcorr_cwave, newmag, newphase)
    return final_xcorr_cwave
