#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from Channel import Channel
from Mapper import Mapper
import numpy as np
import matplotlib.pyplot as plt

def simulate(EbN0, mapper, channel, n_bits):
    # Define BER array
    BER = np.ones(EbN0.shape)

    # Run iterations as threads
    for i in range(len(EbN0)):
        b = mapper.generate_bits(n_bits)
        x = mapper.map_bits(b)
        channel.set_awgn_channel(True, 'bpsk', EbN0[i])
        y, cc, nV = channel.filter_symbols(x)
        b_rx = mapper.demap_symbols(y)
        BER[i] = 1 - np.count_nonzero(b == b_rx) / len(b)

    return BER
    
# Parameters
mod_scheme = 'bpsk'
awgn_enable = True
lti_enable = True
lti_channel_type = 'TYPE3'
n_bits = int(1e6)
EbN0 = np.arange(1, 10, 1)

# Block config
mapper = Mapper(mod_scheme)
channel = Channel()
channel.set_lti_channel(lti_enable, lti_channel_type)

# Simulation
BER = simulate(EbN0, mapper, channel, n_bits)

# Plot BER
plt.figure()
plt.semilogy(EbN0, BER)
plt.show()