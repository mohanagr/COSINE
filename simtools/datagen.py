import numpy as np

class Spectra():
    def __init__():
        pass
    
    def populate():
        pass
    
    def channelize():
        pass
    
    def quantize():
        pass
    
    def simulate():
        pass

def get_samples(delay, mult, k, dk, N=4096, plot=False):
    np.random.seed(42)
    N = N * mult
    dk = int(dk * mult)
    num_k = len(k)
    rr = np.sqrt(100) * np.random.randn(dk * num_k)
    ir = np.sqrt(100) * np.random.randn(dk * num_k)
    fr = np.zeros(N // 2 + 1)
    fi = np.zeros(N // 2 + 1)
    for i, j in enumerate(k):
        j = int(j * mult)
        fr[j - dk // 2 : j + dk // 2] = rr[i * dk : (i + 1) * dk]
        fi[j - dk // 2 : j + dk // 2] = ir[i * dk : (i + 1) * dk]

    x_new = np.fft.irfft(
        fr + np.random.randn(N // 2 + 1) + 1j * (fi + np.random.randn(N // 2 + 1))
    )
    x2_new = np.fft.irfft(
        (fr + 1j * fi) * np.exp(-2j * np.pi * np.arange(0, N // 2 + 1) * delay / N)
        + np.random.randn(N // 2 + 1)
        + 1j * np.random.randn(N // 2 + 1)
    )

    if plot:
        plt.plot(np.abs(np.fft.fft(x2_new)))
    return x_new, x2_new
