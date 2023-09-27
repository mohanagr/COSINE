import numpy as np
import configparser
from utils import prng, pfft


@nb.njit(parallel=True)
def init_arr_1d(arr, val):
    n = arr.shape[0]
    for i in nb.prange(n):
        arr[i] = val


@nb.njit(parallel=True)
def add_noise_to_complex(arr, noise):
    n = arr.shape[0]
    for i in nb.prange(n):
        arr[i] += noise[2 * i] + 1j * noise[2 * i + 1]


@nb.njit(parallel=True)
def apply_delay(arr, delay):
    n = arr.shape[0]
    N = 2 * (n - 1)  # n = N//2 + 1
    for k in nb.prange(n):
        arr[i] = arr[i] * np.exp(-2j * np.pi * k * delay / N)


class Spectra:
    def __init__(path_to_config):
        config = config.ConfigParser()
        config.read(path_to_config)
        self.acclens = [int(x) for x in config["DEFAULT"]["acclens"].strip().split(",")]
        self.niter = int(config["DEFAULT"]["acclens"].strip())
        self.NPFB = int(config["DEFAULT"]["NPFB"].strip())
        self.bandwidth = float(config["DEFAULT"]["bandwidth"].strip())
        self.delay = float(config["DEFAULT"]["delay"].strip())
        self.SNR = float(config["DEFAULT"]["SNR"].strip())
        self.loglevel = int(config["DEFAULT"]["loglevel"].strip())
        self.mult = int(config["DEFAULT"]["mult"].strip()) + 3  # ntap-1
        self.channels = np.asarray([], dtype="int")
        for c_range in config["DEFAULT"]["channels"].strip().split():
            st, en = [int(chan) for chan in c_range.split(":")]
            self.channels = np.hstack([self.channels, np.arange(st, en)])
        self.freqs = self.channels + float(config["DEFAULT"]["chan_offset"].strip())
        self.x1 = None
        self.x2 = None
        self.f1 = None
        self.f2 = None

    def populate(self):
        N = self.NPFB * self.mult  # length of timestream in samples
        dk = int(self.bandwidth * self.mult)
        num_k = len(self.freqs)
        rng = prng.MultithreadedRNG(2 * dk * num_k + 4 * (N // 2 + 1))
        rng.fill()
        rns = rng.values
        ff = np.empty(N // 2 + 1, dtype="complex128")
        init_arr_1d(f, 0)
        for i, k in enumerate(self.chans):
            k = int(k * mult)
            block = ff[k - dk // 2 : k + dk // 2]
            ii = 2 * i
            block[:] = np.sqrt(SNR) * (
                rns[ii * dk : (ii + 1) * dk] + 1j * rns[(ii + 1) * dk : (ii + 2) * dk]
            )
        ff2 = ff.copy()
        start = 2 * num_k * dk
        size = N + 2
        add_noise_to_complex(ff, rns[start : start + size])
        self.x1 = pfft.irfft_1d(ff)
        apply_delay(ff2, delay)
        start += size
        add_noise_to_complex(ff2, rns[start : start + size])
        self.x2 = pfft.irfft_1d(ff2)
        # if plot:
        #     plt.plot(np.abs(np.fft.fft(x2_new)))

    def channelize(self):
        self.f1 = pfft.cpfb(self.x1, nchan=self.NPFB//2)
        self.f2 = pfft.cpfb(self.x2, nchan=self.NPFB//2)

    def quantize():
        pass

    def simulate():
        pass
        
