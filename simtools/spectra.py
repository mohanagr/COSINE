import numpy as np
import configparser
from .utils import prng, pfft
import numba as nb
import time
from matplotlib import pyplot as plt

@nb.njit(parallel=True)
def init_arr_1d(arr, val):
    n = arr.shape[0]
    for i in nb.prange(n):
        arr[i] = val


@nb.njit(parallel=True)
def add_noise_to_complex(arr, noise):
    n = arr.shape[0]
    # print("n is", n)
    # print("first two noise elements", noise[0], noise[1])
    for i in nb.prange(n):
        arr[i] += noise[2 * i] + 1j * noise[2 * i + 1]


@nb.njit(parallel=True)
def apply_delay(arr, delay):
    n = arr.shape[0]
    N = 2 * (n - 1)  # n = N//2 + 1
    for k in nb.prange(n):
        arr[k] = arr[k] * np.exp(-2j * np.pi * k * delay / N)


class Spectra:
    def __init__(self, npfb, nspec, channels, bandwidth, delay, snr, chan_offset):
        
        self.npfb = npfb
        self.channels =  channels
        self.channels.sort()
        self.bandwidth = bandwidth
        self.delay = delay
        self.snr = snr
        self.nspec = nspec
        self.mult = nspec
        self.chan_offset = 0.1
        self.freqs = self.channels + self.chan_offset
        self.x1 = None
        self.x2 = None
        self.f1 = None
        self.f2 = None

    def populate(self):
        N = self.npfb * self.mult  # length of timestream in samples
        dk = int(self.bandwidth * self.mult)
    #     print(dk)
        num_k = len(self.freqs)
    #     print("starting to generate rng...")
    #     t1=time.time()
        rng = prng.MultithreadedRNG(2 * dk * num_k + 4 * (N // 2 + 1))
        rng.fill()
        rns = rng.values
        # rns = np.random.randn(2 * dk * num_k + 4 * (N // 2 + 1))
        # print("Rns shape pop1", rns.shape)
    #     t2=time.time()
    #     print("took ",t2-t1, "for rng")
        ff = np.empty(N // 2 + 1, dtype="complex128")
        init_arr_1d(ff, 0)
    #     print("ff init",np.sum(ff))
    #     print(self.freqs)
        for i, k in enumerate(self.freqs):
            k = int(k * self.mult)
    #         # print("k = ", k)
    #         block = ff[k - dk // 2 : k + dk // 2]
            ii = 2 * i
    #         # print(ii)
    #         print("block is from", k - dk // 2, k + dk // 2)
    #         print("using rns from:", ii*dk, (ii+1)*dk,(ii + 1) * dk, (ii + 2) * dk )
            ff[k - dk // 2 : k + dk // 2] = np.sqrt(self.snr) * (
                rns[ii * dk : (ii + 1) * dk] + 1j * rns[(ii + 1) * dk : (ii + 2) * dk]
            )
        ff2 = ff.copy()
    #     # plt.plot(np.abs(ff))
    #     # plt.show()
        start = 2 * num_k * dk
        # print("start after fill", start)
        size = N+2
    #     print("start is ", start)
    #     print("size is", size)
    #     print("next two noise elements", rns[start], rns[start+1])
        add_noise_to_complex(ff, rns[start : start + size])
        apply_delay(ff2, self.delay)
        start += size
        add_noise_to_complex(ff2, rns[start : start + size])
        # plt.plot(np.abs(ff2))
        # plt.plot(np.abs(ff))
        # plt.show()
        self.x1 = pfft.irfft_1d(ff)
        self.x2 = pfft.irfft_1d(ff2)
        

    def channelize(self):
        self.f1 = pfft.cpfb(self.x1, nchan=self.npfb//2)
        self.f2 = pfft.cpfb(self.x2, nchan=self.npfb//2)

    def quantize():
        pass

    def generate(self):
        with pfft.parallelize_fft():
            #t1=time.time()
            self.populate()
            #t2=time.time()
            #print("generation took:", t2-t1)
            self.channelize()
        
