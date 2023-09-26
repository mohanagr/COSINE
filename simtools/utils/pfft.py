import ctypes
import numpy as np
import os
from contextlib import contextmanager

mylib = ctypes.cdll.LoadLibrary(os.path.realpath(__file__ + r"/..") + "/libpfb.so")

# mylib.myfft.argtypes = [
#     ctypes.c_void_p,
#     ctypes.c_void_p,
#     ctypes.c_int64
# ]

mylib.test.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64]

mylib.pfb.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_int64,
]

mylib.set_threads.argtypes = [ctypes.c_int]


# def fft(inp):
#     n=inp.shape[0]
#     output=np.empty(n,dtype="complex128",order="c")
#     mylib.myfft(inp.ctypes.data,output.ctypes.data,n)
#     return output


@contextmanager
def parallel_fft(nthreads):
    mylib.set_threads(nthreads)
    yield
    mylib.cleanup_threads()

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


def very_fast_pfb(timestream, nchan=2048, ntap=4, window=sinc_hanning):
    lblock = 2 * (nchan)
    w = window(ntap, lblock)
    # w = np.ones(ntap*lblock,dtype="float64",order="c")
    nspec = timestream.size // lblock - (ntap - 1)
    # spec=np.zeros((nspec,lblock))
    # for bi in range(nspec):
    #     # cut out the correct timestream section
    #     spec[bi] = np.sum((timestream[bi*lblock:(bi+ntap)*lblock]*w).reshape(ntap,lblock),axis=0)
    # test_op = np.fft.rfft(spec,axis=1)
    if nspec == int(nspec):
        nspec = int(nspec)
    else:
        raise Exception("nspec is {}, should be integer".format(nspec))
    output = np.empty((nspec, lblock // 2 + 1), dtype="complex128", order="c")
    mylib.pfb(
        timestream.ctypes.data, output.ctypes.data, w.ctypes.data, nspec, nchan, ntap
    )
    # print(test_op-output)
    return output

nb.config.THREADING_LAYER_PRIORITY = ["omp", "tbb", "workqueue"]


def sinc_hanning(ntap, lblock):
    N = ntap * lblock
    w = np.arange(0, N) - N / 2
    return np.hanning(ntap * lblock) * np.sinc(w / lblock)


def slow_pfb(timestream, nchan=2048, ntap=4, window=sinc_hanning):
    # number of samples in a sub block
    lblock = 2 * (nchan)
    # number of blocks
    nblock = timestream.size / lblock - (ntap - 1)
    if nblock == int(nblock):
        nblock = int(nblock)
    else:
        raise Exception("nblock is {}, should be integer".format(nblock))

    # initialize array for spectrum
    spec = np.zeros((nblock, 2 * nchan), dtype=np.complex128)

    # window function
    w = window(ntap, lblock)

    def s(ts_sec):
        return np.sum(
            ts_sec.reshape(ntap, lblock), axis=0
        )  # this is equivalent to sampling an ntap*lblock long fft - M

    # iterate over blocks and perform PFB
    for bi in range(nblock):
        # cut out the correct timestream section
        ts_sec = timestream[bi * lblock : (bi + ntap) * lblock].copy()

        spec[bi] = np.fft.fft(s(ts_sec * w))

    return spec


def fast_pfb(timestream, nchan=2048, ntap=4, window=sinc_hanning):
    # number of samples in a sub block
    lblock = 2 * (nchan)
    # number of blocks
    nblock = timestream.size / lblock - (ntap - 1)
    if nblock == int(nblock):
        nblock = int(nblock)
    else:
        raise Exception("nblock is {}, should be integer".format(nblock))

    # initialize array for spectrum
    spec = np.zeros((nblock, 2 * nchan), dtype="float64")
    w = window(ntap, lblock)
    fill_blocks(spec, timestream, w, nblock, lblock, ntap)
    with fft.set_workers(40):
        spec = fft.rfft(spec, axis=1)
    return spec


@nb.njit(parallel=True)
def fill_blocks(spec, timestream, window, nblock, lblock, ntap):
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
