import numpy as np
import numba as nb
from scipy import fft
import multiprocessing as mp
# from parallel_random import MultithreadedRNG
import time
nb.config.THREADING_LAYER_PRIORITY = ["omp", "tbb", "workqueue"]
def sinc_hanning(ntap,lblock):
    N=ntap*lblock
    w=np.arange(0,N)-N/2
    return np.hanning(ntap*lblock)*np.sinc(w/lblock)

def forward_pfb(timestream, nchan=2048, ntap=4, window=sinc_hanning):

    # number of samples in a sub block
    lblock = 2*(nchan)
    # number of blocks
    nblock = timestream.size / lblock - (ntap - 1)
    if nblock==int(nblock): nblock=int(nblock)
    else: raise Exception("nblock is {}, should be integer".format(nblock))

    # initialize array for spectrum 
    spec = np.zeros((nblock,2*nchan), dtype=np.complex128)

    # window function
    w = window(ntap, lblock)

    def s(ts_sec):
        return np.sum(ts_sec.reshape(ntap,lblock),axis=0) # this is equivalent to sampling an ntap*lblock long fft - M


    # iterate over blocks and perform PFB
    for bi in range(nblock):
        # cut out the correct timestream section
        ts_sec = timestream[bi*lblock:(bi+ntap)*lblock].copy()

        spec[bi] = np.fft.fft(s(ts_sec * w)) 

    return spec

def forward_pfb2(timestream, nchan=2048, ntap=4, window=sinc_hanning):

    # number of samples in a sub block
    lblock = 2*(nchan)
    # number of blocks
    nblock = timestream.size / lblock - (ntap - 1)
    if nblock==int(nblock): nblock=int(nblock)
    else: raise Exception("nblock is {}, should be integer".format(nblock))

    # initialize array for spectrum 
    spec = np.zeros((nblock,2*nchan), dtype="float64")

    # window function
    w = window(ntap, lblock)

#     # iterate over blocks and perform PFB
#     for bi in range(nblock):
#         # cut out the correct timestream section
#         spec[bi] = np.fft.fft(np.sum((timestream[bi*lblock:(bi+ntap)*lblock]*w).reshape(ntap,lblock),axis=0)) 
    t1=time.time()
    fill_blocks(spec,timestream,w,nblock,lblock,ntap)
    t2=time.time()
    #print("time taken to set things up", t2-t1)
    t1=time.time()
    with fft.set_workers(40):
        spec=fft.rfft(spec,axis=1)
    t2=time.time()
    #print("time for fft",t2-t1)
    return spec

@nb.njit(parallel=True)
def fill_blocks(spec,timestream,window,nblock,lblock,ntap):
    for bi in nb.prange(nblock):
        spec[bi] = np.sum((timestream[bi*lblock:(bi+ntap)*lblock]*window).reshape(ntap,lblock),axis=0)
    # for i in range(nblock):
    #     for j in range(lblock):
    #         for k in range(ntap):
    #             spec[i*lblock+j]+=timestream[(i+k)*lblock+j]*window[k*lblock+j]

if __name__=="__main__":
	mrng = MultithreadedRNG(1000003*4096, seed=12345)
	mrng.fill()
	print("generated randoms. starting pfb")
	niter=100
	#t1=time.time()
	#for i in range(niter):
	#	y=forward_pfb(mrng.values)
	#t2=time.time()
	#print("normal version per iteration",(t2-t1)/niter)
	t1=time.time()
	for i in range(niter):
		y=forward_pfb2(mrng.values)
	t2=time.time()
	print("parallel version per iteration",(t2-t1)/niter)
