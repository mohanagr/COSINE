import numpy as np
import time
import numba as nb
from simtools.utils import pfft, prng

if __name__ == "__main__":
    rng = prng.MultithreadedRNG(100003 * 4096)
    rng.fill()
    x = rng.values
    lblock = 4096
    # x = np.ones(lblock*5,dtype="float64", order="c")
    # x=np.tile(np.arange(lblock),7).astype("float64")
    print("starting test...")
    niter = 1
    t1 = time.time()
    with pfft.parallelize_fft(8):
        for i in range(niter):
            y1 = pfft.cpfb(x, nchan=lblock // 2)
            # y1= pfft.rfft_1d(x)
    t2 = time.time()
    print("C version per iteration", (t2 - t1) / niter)

    t1 = time.time()
    for i in range(niter):
        y2 = pfft.pypfb(x, nchan=lblock // 2, fast=True)
    t2 = time.time()
    print("python version per iteration", (t2 - t1) / niter)
    print("max error two pfbs", np.max(np.std(np.real(y2) - np.real(y1))))
    # print(y1)
    # print(y2)

    # x = np.random.randn(1024*1024)
    # t1=time.time()
    # fft1=pfft.rfft_1d(x)
    # t2=time.time()
    # print("parallel fftw", t2-t1)

    # t1=time.time()
    # fft2=np.fft.rfft(x)
    # t2=time.time()
    # print("np fft", t2-t1)
    # print("max error real of ffts", np.max(np.abs(np.real(fft1)-np.real(fft2))))
    # print("max error imag of ffts", np.max(np.abs(np.imag(fft1)-np.imag(fft2))))
    # x1 = np.fft.irfft(fft2)
    # x2 = pfft.irfft_1d(fft1)
    # print("x1 acc", np.std(x1-x))
    # print("x2 acc", np.std(x2-x))