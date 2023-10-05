from simtools.spectra import Spectra
import configparser
import numpy as np

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
    nspec = int(config["DEFAULT"]["nspec"].strip())
    chan_offset = float(config["DEFAULT"]["offset"].strip())
    channels = np.asarray([], dtype="int")
    for c_range in config["DEFAULT"]["channels"].strip().split():
        st, en = [int(chan) for chan in c_range.split(":")]
        channels = np.hstack([channels, np.arange(st, en)])

    print(npfb, acclens, channels, bandwidth, delay, snr, nspec, chan_offset)
    myspec = Spectra(npfb, acclens[0], channels, bandwidth, delay, snr, nspec, chan_offset)
    for i in range(1):
        myspec.generate()
