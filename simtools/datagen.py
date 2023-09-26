import numpy as np
import configparser
from utils import prng, pfft

@nb.njit(parallel=True)
def init_arr_1d(arr,val):
    n=arr.shape[0]
    for i in nb.prange(n):
        arr[i] = val

@nb.njit(parallel=True)
def add_noise_to_complex(arr,noise):
    n=arr.shape[0]
    for i in nb.prange(n):
        arr[i] = arr[i].real + noise[2*i] + 1j * (arr[i].imag + noise[2*i+1])



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

    def populate():
        pass

    def channelize():
        pass

    def quantize():
        pass

    def simulate():
        pass

    def _generate_samples(self):
        N = self.NPFB * self.mult #length of timestream in samples
        dk = int(self.bandwidth * self.mult)
        num_k = len(self.freqs)
        rng = prng.MultithreadedRNG(2 * dk * num_k + 4 * (N // 2 + 1))
        rng.fill()
        rns = rng.values
        fr = np.empty(N // 2 + 1)
        fi = np.empty(N // 2 + 1)
        bb = num_k * dk
        for i, j in enumerate(k):
            j = int(j * mult)
            fr[j - dk // 2 : j + dk // 2] = np.sqrt(self.SNR) * rns[i * dk : (i + 1) * dk]
            fi[j - dk // 2 : j + dk // 2] = (
                np.sqrt(self.SNR) * rns[bb + i * dk : bb + (i + 1) * dk]
            )
        bb += num_k * dk
        size = N // 2 + 1
        #     x_new=np.fft.irfft(fr+np.random.randn(N//2+1)+1J*(fi+np.random.randn(N//2+1)))
        x_new = pfft.irfft_1d(
            fr + rns[bb : bb + size] + 1j * (fi + rns[bb + size : bb + 2 * size])
        )
        bb += 2 * size
        x2_new = pfft.irfft_1d(
            (fr + 1j * fi) * np.exp(-2j * np.pi * np.arange(0, N // 2 + 1) * delay / N)
            + rns[bb : bb + size]
            + 1j * rns[bb + size : bb + 2 * size]
        )
        if plot:
            plt.plot(np.abs(np.fft.fft(x2_new)))
        return x_new, x2_new
