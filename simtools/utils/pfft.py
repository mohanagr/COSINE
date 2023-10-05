import ctypes
import numpy as np
import os
from contextlib import contextmanager
import numba as nb
from scipy import fft

nb.config.THREADING_LAYER_PRIORITY = ["omp", "tbb", "workqueue"]
NUM_CPU = os.cpu_count()

mylib = ctypes.cdll.LoadLibrary(os.path.realpath(__file__ + r"/..") + "/libpfb.so")

mylib.test.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64]

mylib.pfb.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_int64,
]

mylib.fft_c2r_1d.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int64,
    ctypes.c_int,
]

mylib.fft_r2c_1d.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int64,
]

mylib.set_threads.argtypes = [ctypes.c_int]


@contextmanager
def parallelize_fft(nthreads):
    mylib.set_threads(nthreads)
    yield
    mylib.cleanup_threads()


def rfft_1d(dat):
    n = dat.shape[0]
    datft = np.empty(n // 2 + 1, dtype="complex128", order="c")
    mylib.fft_r2c_1d(dat.ctypes.data, datft.ctypes.data, n)
    return datft


def irfft_1d(datft, preserve_input=0):
    if datft.shape[0] % 2 == 0:
        raise ValueError("you're doing a complex to real rfft. n should've been odd?")
    n = 2 * (datft.shape[0] - 1)
    dat = np.empty(n, dtype="float64", order="c")
    mylib.fft_c2r_1d(datft.ctypes.data, dat.ctypes.data, n, preserve_input)
    return dat


def mytest():
    n = 4
    # inp = np.zeros(n,dtype="complex128",order='c')
    #  inp[0] = 100. + 1J*200.
    #  inp[1] = 300. + 1J*500.
    inp = np.tile(np.arange(0, 4), 2).astype("float64")
    inp = inp.reshape(2, 4)
    inp = np.ones((2, 4), dtype="float64", order="c")
    print(inp)
    output = np.zeros((2, 3), dtype="complex128", order="c")
    mylib.test(inp.ctypes.data, output.ctypes.data, 2, 4)
    print(output)
    print(np.fft.fft(inp, axis=1))


def sinc_hanning(ntap, lblock):
    N = ntap * lblock
    w = np.arange(0, N) - N / 2
    return np.hanning(ntap * lblock) * np.sinc(w / lblock)


def cpfb(timestream, nchan=2048, ntap=4, window=sinc_hanning):
    lblock = 2 * (nchan)
    w = window(ntap, lblock)
    nspec = timestream.size // lblock - (ntap - 1)
    if nspec == int(nspec):
        nspec = int(nspec)
    else:
        raise Exception("nspec is {}, should be integer".format(nspec))
    output = np.empty((nspec, nchan + 1), dtype="complex128", order="c")
    mylib.pfb(
        timestream.ctypes.data, output.ctypes.data, w.ctypes.data, nspec, nchan, ntap
    )
    return output


def sinc_hanning(ntap, lblock):
    N = ntap * lblock
    w = np.arange(0, N) - N / 2
    return np.hanning(ntap * lblock) * np.sinc(w / lblock)


def pypfb(timestream, nchan=2048, ntap=4, window=sinc_hanning, fast=False):
    # number of samples in a sub block
    lblock = 2 * (nchan)
    # number of blocks
    nblock = timestream.size / lblock - (ntap - 1)
    if nblock == int(nblock):
        nblock = int(nblock)
    else:
        raise Exception("nblock is {}, should be integer".format(nblock))

    # initialize array for spectrum
    spec = np.zeros((nblock, lblock), dtype="float64")

    # window function
    w = window(ntap, lblock)

    if fast:
        fill_blocks(spec, timestream, w, nblock, lblock, ntap)
        with fft.set_workers(
            NUM_CPU // 2
        ):  # generally os.cpu_count() returns 2x physical cores
            spec = fft.rfft(spec, axis=1)
    else:
        # print(spec.shape, lblock)
        for bi in range(nblock):
            spec[bi,:] = np.sum((timestream[bi * lblock : (bi + ntap) * lblock] * w).reshape(ntap, lblock),axis=0)
        spec = np.fft.rfft(spec,axis=1)
    return spec


@nb.njit(parallel=True)
def fill_blocks(spec, timestream, window, nblock, lblock, ntap):
    #print(nblock, lblock, ntap, timestream.shape, spec.shape)
    for bi in nb.prange(nblock):
        spec[bi] = np.sum(
            (timestream[bi * lblock : (bi + ntap) * lblock] * window).reshape(
                ntap, lblock
            ),
            axis=0,
        )


if __name__ == "__main__":
    import time

    # x=np.random.randn(1000000) + 1J*np.random.randn(1000000)
    # niter=1
    # t1=time.time()
    # for i in range(1):
    #     y=fft(x)
    # t2=time.time()
    # print("fftw parallel",(t2-t1)/1000)
    # t1=time.time()
    # for i in range(1):
    #     y1=np.fft.fft(x)
    # t2=time.time()
    # print("fft np",(t2-t1)/1000)
    # print(np.mean(np.abs(y-y1)))
    # print(np.mean)
    # lblock=4
    # timestream = np.ones(lblock*13,dtype="float64", order="c")
    # timestream = np.tile(np.arange(lblock),13).astype("float64")
    # print(timestream.shape)
    # op=fast_pfb(timestream,nchan=2)
    # print("pfb", op)

    # print(np.fft.rfft(timestream[:10]))
    # print("timestream",timestream.reshape(-1,lblock))
    # op2 = np.fft.rfft(timestream.reshape(-1,lblock),axis=1)
    # print("python", op2)
    # print(op)
    # mytest()
