from utils.fftdriver import fast_pfb
from utils.parallel_pfb import forward_pfb, forward_pfb2
from utils.parallel_random import MultithreadedRNG
import numpy as np
import time
import numba as nb
#nb.config.THREADING_LAYER="omp"
nb.config.THREADING_LAYER_PRIORITY = ["omp", "tbb", "workqueue"]
if __name__=="__main__":
    #x=np.random.randn(100003*4096)
    rng=MultithreadedRNG(100003*4096)
    rng.fill()
    x=rng.values
    lblock=4096
    # x = np.ones(lblock*5,dtype="float64", order="c")
    #x=np.tile(np.arange(lblock),7).astype("float64")
    print("starting test...")
    niter=100
    t1=time.time()
    for i in range(niter):
        y1=fast_pfb(x, nchan=lblock//2)
    t2=time.time()
    print("normal version per iteration",(t2-t1)/niter)
    
    t1=time.time()
    for i in range(niter):
        y2=forward_pfb2(x,nchan=lblock//2)
    t2=time.time()
    print("parallel version per iteration",(t2-t1)/niter)
    print(np.max(np.abs(y2[:,:lblock//2+1]-y1)))
    # print(y1)
    # print(y2)
    
