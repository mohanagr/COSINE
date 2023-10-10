from simtools.spectra import Spectra
import configparser
import numpy as np
from matplotlib import pyplot as plt
from simtools.utils import fftmod
import time

def run_sim(spectra, osamp=1):
    xcorrtime2 = fftmod.get_coarse_xcorr(spectra.f1,spectra.f2,spectra.channels)
    xcorrtime2=np.fft.fftshift(xcorrtime2, axes=1)
    N=xcorrtime2.shape[1]
    dN=int(0.1*N)
    stamp=slice(N//2-dN,N//2+dN)
    # print("ABLE TO ACCESS MYSPEC",myspec)
    xcorr2sum = np.sum(xcorrtime2,axis=0)
    # print(np.argmax(xcorr2sum))
    # plt.title(f"Max at {np.argmax(xcorr2sum[stamp])-dN}")
    # plt.plot(xcorr2sum[stamp].real)
    # plt.show()

    # dN=200
    # stamp=slice(N//2-dN,N//2+dN)
    # osamp=10
    # xcorr_arr = np.zeros((len(spectra.channels),2*dN*4096*osamp),dtype='complex128')
    # sample_no=np.arange(0,2*dN*4096*osamp)
    # coarse_sample_no=np.arange(0,2*dN)*4096*osamp
    # # print("osamp passed is",osamp)
    # t1=time.time()
    # for i, chan in enumerate(spectra.channels):    
    #     xcorr_arr[i,:] = fftmod.get_interp_xcorr(xcorrtime2[i,stamp],chan,sample_no,coarse_sample_no)
    # t2=time.time()
    # print(t2-t1,(t2-t1)/9)
    return xcorr2sum

if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read("./config.ini")
    acclens = [int(x) for x in config["DEFAULT"]["acclens"].strip().split(",")]
    niter = int(config["DEFAULT"]["niter"].strip())
    npfb = int(config["DEFAULT"]["npfb"].strip())
    bandwidth = float(config["DEFAULT"]["bandwidth"].strip())
    delay = float(config["DEFAULT"]["delay"].strip())
    snr = float(config["DEFAULT"]["snr"].strip())
    loglevel = int(config["DEFAULT"]["loglevel"].strip())
    chan_offset = float(config["DEFAULT"]["offset"].strip())
    channels = np.asarray([], dtype="int")
    for c_range in config["DEFAULT"]["channels"].strip().split():
        st, en = [int(chan) for chan in c_range.split(":")]
        channels = np.hstack([channels, np.arange(st, en)])

    print(npfb, acclens, channels, bandwidth, delay, snr, chan_offset)
    f=plt.gcf()
    f.set_size_inches(10,4)
    niter=10
    peak_heights = np.zeros(niter)
    coarse_peak_heights = np.zeros(niter)
    num_conf_peaks = np.zeros(niter)
    for i in range(niter):
        print("ON ITERATION", i+1)
        myspec = Spectra(npfb, acclens[0], channels, bandwidth, delay, snr, chan_offset)
        myspec.generate()
        # print("spectra shapes", myspec.f1.shape, myspec.f2.shape)
        # plt.subplot(121)
        # plt.imshow(np.log10(np.abs(myspec.f1[:,1834:1854])),aspect="auto",interpolation="none")
        # plt.subplot(122)
        # plt.imshow(np.log10(np.abs(myspec.f2[:,1834:1854])),aspect="auto",interpolation="none")
        # plt.tight_layout()
        # plt.show()
        xcorr2sum = run_sim(myspec, osamp=10)
        coarse_m = -122 + len(xcorr2sum)//2
        peak_heights[i] = xcorr2sum[coarse_m]
        # final_xcorr = np.sum(xcorr_arr,axis=0)
        # m=np.argmax(final_xcorr.real)
        # print("max fine xcorr", m-len(final_xcorr)//2)
        # true_m = -5000004+len(final_xcorr)//2
        # print("m vs true_m", m, true_m)
        # #track height of coarse peak
        # coarse_m = -122*4096*10+len(final_xcorr)//2
        # peak_heights[i] = final_xcorr.real[coarse_m]
        # coarse_m = -122 + len(xcorr2sum)//2
        # coarse_peak_heights[i] = xcorr2sum.real[coarse_m]
        # sigma=2e-6
        # true_height = final_xcorr.real[true_m]
        # num_conf_peaks[i] = len(np.where(np.logical_and(final_xcorr.real>(true_height-sigma),final_xcorr.real<(true_height+sigma)))[0])
        # diff=1<<12
        # print(diff)
        # for i in range(0, len(myspec.channels)):
        #     for j in range(i+1, len(myspec.channels)):
        #         temp=myspec.channels[j]-myspec.channels[i]
        #         if(temp<diff):
        #             diff=temp
        # diff=diff/2 #beat frequency is diff/2
        # print("diff is", diff)
        # lims = int(npfb*10/diff/2) #limit is -T/2 T/2
        # print("plot limits are", -lims,lims)
        # lims=200000
        # plt.clf()
        # plt.plot(final_xcorr.real[m-lims:m+lims],"-*")
        # plt.show()
    
    print("peak_heights are", peak_heights)
    # print("coarse peak heights are", coarse_peak_heights)
    print("spread", np.std(peak_heights))
    # print("number of confusing peaks", num_conf_peaks)
