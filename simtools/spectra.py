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


@nb.njit()
def add_noise_to_complex(arr, noise):
    n = arr.shape[0]
    print("n is", n)
    print("first two noise elements", noise[0], noise[1])
    for i in nb.prange(n):
        arr[i] += noise[2 * i] + 1j * noise[2 * i + 1]


@nb.njit(parallel=True)
def apply_delay(arr, delay):
    n = arr.shape[0]
    N = 2 * (n - 1)  # n = N//2 + 1
    for k in nb.prange(n):
        arr[k] = arr[k] * np.exp(-2j * np.pi * k * delay / N)


class Spectra:
    def __init__(self, npfb, acclen, channels, bandwidth, delay, snr, nspec, chan_offset):
        
        self.npfb = npfb
        self.acclen = acclen
        self.channels =  channels
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
        print(dk)
        num_k = len(self.freqs)
        print("starting to generate rng...")
        t1=time.time()
        # rng = prng.MultithreadedRNG(2 * dk * num_k + 4 * (N // 2 + 1))
        # rng.fill()
        # rns = rng.values
        rns = np.random.randn(2 * dk * num_k + 4 * (N // 2 + 1))
        t2=time.time()
        print("took ",t2-t1, "for rng")
        ff = np.zeros(N // 2 + 1, dtype="complex128")
        # init_arr_1d(ff, 0)
        print("ff init",np.sum(ff))
        print(self.freqs)
        for i, k in enumerate(self.freqs):
            k = int(k * self.mult)
            # print("k = ", k)
            block = ff[k - dk // 2 : k + dk // 2]
            ii = 2 * i
            # print(ii)
            print("block is from", k - dk // 2, k + dk // 2)
            print("using rns from:", ii*dk, (ii+1)*dk,(ii + 1) * dk, (ii + 2) * dk )
            block[:] = np.sqrt(self.snr) * (
                rns[ii * dk : (ii + 1) * dk] + 1j * rns[(ii + 1) * dk : (ii + 2) * dk]
            )
        ff2 = ff.copy()
        start = 2 * num_k * dk
        size = N + 2
        print("start is ", start)
        print("size is", size)
        print("next two noise elements", rns[start], rns[start+1])
        # add_noise_to_complex(ff, rns[start : start + size])
        plt.plot(rns[start : start + size])
        plt.show()
        fr = ff.real + rns[start : start + N//2+1]
        fi = ff.imag + rns[start+N//2+1:start+N+2]
        ff = fr + 1j*fi
        # ff[:] = ff + rns[start : start + N//2+1] + 1J*rns[start+N//2+1:start+N+2]
        self.x1 = pfft.irfft_1d(ff)
        apply_delay(ff2, self.delay)
        start += size
        add_noise_to_complex(ff2, rns[start : start + size])
        self.x2 = pfft.irfft_1d(ff2)
        # if plot:
        plt.clf()
        plt.plot(np.abs(ff))
        plt.show()
    
    # def populate(self):
    #     N=self.npfb*self.mult
    #     dk=int(self.bandwidth*self.mult)
    #     num_k = len(self.freqs)
    #     t1=time.time()
    #     rng = prng.MultithreadedRNG(2*dk*num_k+4*(N//2+1))
    #     rng.fill()
    #     rns = rng.values
    # #     rns=np.random.randn(2*dk*num_k+4*(N//2+1))
    #     t2=time.time()
    #     print("rng:",t2-t1)

    # #     rns=np.random.randn(2*dk*num_k+4*(N//2+1))
    # #     rr=np.sqrt(100)*np.random.randn(dk*num_k)
    # #     ir=np.sqrt(100)*np.random.randn(dk*num_k)
    #     fr=np.zeros(N//2+1)
    #     fi=np.zeros(N//2+1)
    #     bb=num_k*dk
    #     for i,j in enumerate(self.freqs):
    #         j= int(j*self.mult)
    #         fr[j-dk//2:j+dk//2]=np.sqrt(self.snr)*rns[i*dk:(i+1)*dk]
    #         fi[j-dk//2:j+dk//2]=np.sqrt(self.snr)*rns[bb+i*dk:bb+(i+1)*dk]
    #     bb+=num_k*dk
    #     size=N//2+1
    # #     x_new=np.fft.irfft(fr+np.random.randn(N//2+1)+1J*(fi+np.random.randn(N//2+1)))
    #     x_new=np.fft.irfft(fr+rns[bb:bb+size]+1J*(fi+rns[bb+size:bb+2*size]))

    # #     f2=(fr+1J*fi)*np.exp(-2J*np.pi*np.arange(0,N//2+1)*delay/N)
    #     bb+=2*size
    #     x2_new = np.fft.irfft((fr+1J*fi)*np.exp(-2J*np.pi*np.arange(0,N//2+1)*self.delay/N) + rns[bb:bb+size]+1J*rns[bb+size:bb+2*size])
    #     plt.plot(rns[0:1000000])
    #     plt.show()

    def channelize(self):
        self.f1 = pfft.cpfb(self.x1, nchan=self.npfb//2)
        self.f2 = pfft.cpfb(self.x2, nchan=self.npfb//2)

    def quantize():
        pass

    def generate(self):
        # with pfft.parallelize_fft(8):
        # t1=time.time()
        self.populate()
        # t2=time.time()
        # print("generation took:", t2-t1)
        # self.channelize()
        
