from extended_ring_model import *
from dict_key_definitions import rmkeys
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    w1 = 0.8
    w2 = 0.2
    rm = RingModel({rmkeys.n_r:8,
                    rmkeys.verbose:True,
                    rmkeys.d_w1:w1,
                    rmkeys.d_w2:w2
    })
    print("= INITIALISATION =")
    rm.initialise(w1=w1,w2=w2)
    print("= INIT COMPLETE =")
    rm.print_spec()


