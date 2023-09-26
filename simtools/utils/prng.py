from numpy.random import default_rng, SeedSequence
import multiprocessing
import concurrent.futures
import numpy as np

class MultithreadedRNG:
    def __init__(self, n, seed=None, threads=None):
        if threads is None:
            threads = multiprocessing.cpu_count()
        self.threads = threads

        seq = SeedSequence(seed)
        self._random_generators = [default_rng(s) for s in seq.spawn(threads)]

        self.n = n
        self.executor = concurrent.futures.ThreadPoolExecutor(threads)
        self.values = np.empty(n)
        self.step = np.ceil(n / threads).astype(np.int_)

    def fill(self):
        def _fill(random_state, out, first, last):
            random_state.standard_normal(out=out[first:last])

        futures = {}
        for i in range(self.threads):
            args = (
                _fill,
                self._random_generators[i],
                self.values,
                i * self.step,
                (i + 1) * self.step,
            )
            futures[self.executor.submit(*args)] = i
        concurrent.futures.wait(futures)

    def __del__(self):
        self.executor.shutdown(False)


if __name__ == "__main__":
    import time
    mrng = MultithreadedRNG(100000000, seed=12345)
    # values = np.empty(100000000)
    # rg = default_rng()
    t1 = time.time()
    for i in range(100000):
        mrng.fill()
    # rg.standard_normal(out=values)
    t2 = time.time()
    print((t2 - t1) / 100)
