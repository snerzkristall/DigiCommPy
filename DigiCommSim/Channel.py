#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

class Channel:
    """
    A class used to simulate AWGN and LTI channels.

    Attributes
    ----------
    awgn_enabled : bool
        Flag for AWGN channel activation
    lti_enabled : bool
        Flag for LTI channel activation
    noise_variance : float
        The noise variance of the AWGN channel
    channel_coeffs : ndarray
        The channel coefficients of the LTI channel

    Methods
    -------
    filter_symbols
        Calculates the resulting symbol vector based on both channels
    set_awgn_channel
        Sets the AWGN channel parameters
    set_lti_channel
        Sets the LTI channel parameters
    get_H
        Calculates the channel matrix (toeplitz)
    """
    
    def __init__(self):
        self.awgn_enabled = False
        self.lti_enabled = False

    def filter_symbols(self, symbol_vector: np.ndarray) -> np.ndarray:
        """
        Filters symbols based on the LTI channel and adds AWGN channel noise.

        Parameters
        ---------
            symbol_vector : ndarray
                The original symbols
            bit_gen_mode: str, optional
                The type of bits to generate (default  is 'random')
            seed: int, optional
                The seed for numpy RNG module (default is '1926')
        
        Returns
        -------
            ndarray
                The filtered symbols
            ndarray
                The LTI channel coefficients
            float
                The AWGN noise variance
        """
        if self.lti_enabled:
            symbol_vector = np.convolve(symbol_vector, self.channel_coeffs)[:-(self.channel_coeffs.size-1)]
        if self.awgn_enabled:
            noise_vector = np.random.normal(0, np.sqrt(self.noise_variance), size=symbol_vector.shape) + 1j*np.random.normal(0, np.sqrt(self.noise_variance), size=symbol_vector.shape)   
        return symbol_vector + noise_vector, self.channel_coeffs, self.noise_variance

    def set_awgn_channel(self, is_enabled: bool, mod_scheme: str, EbN0: int):
        """
        Sets parameters for the AWGN channel
        
        Parameters
        ---------
            is_enabled: bool
                Flag for AWGN channel activation
            mod_scheme : str
                The modulation scheme
            EbN0 : int
                The noise power in dB
        
        Raises
        ------
            NotImplementedError
                If the given bit generation mode is not supported.
        """
        match mod_scheme:
            case 'bpsk':
                M = 2
            case 'qpsk':
                M = 4
            case '16qam':
                M = 16
            case _:
                raise NotImplementedError(f"Modulation scheme {mod_scheme} not supported!")
        self.awgn_enabled = is_enabled
        EbN0_lin = 10**(EbN0/10);    
        self.noise_variance = 1 / (2 * np.log2(M) * EbN0_lin);    

    def set_lti_channel(self, is_enabled: bool, channel_type: str, channel_length: int=5, channel_coeffs: np.ndarray=np.empty(0), seed: int=1926):
        """
        Creates LTI channel coefficients and filter the input symbol sequence.

        Parameters
        ---------
            is_enabled: bool
                Flag for LTI channel activation
            channel_type : str
                The channel type
            channel_length : int
                The number of channel coefficients (default is 5)
            channel_coeffs : ndarray
                The custom channel coefficients (default is None)
            seed : int
                The seed for random channel coefficients (default is 1926)
        
        Raises
        ------
            NotImplementedError
                If the given bit generation mode is not supported.
        """
        match channel_type:
            case 'TYPE1':
                cc = np.array([0.04, -0.05, 0.07, -0.21, -0.5, 0.72, 0.36, 0.21, 0.03, 0.07])
            case 'TYPE2':
                cc = np.array([0.408, 0.816, 0.408])
            case 'TYPE3':
                cc = np.array([0.227, 0.46, 0.688, 0.46, 0.227])
            case 'TYPE4':
                cc = np.empty(channel_length, dtype=complex);
                for idx in range(channel_length):
                    cc[idx] = np.exp(-0.3*1j*idx) + np.exp(-1j*idx)
            case 'TYPE5':
                cc = [2+0.4j, 1.5+1.8j, 1, 1.2-1.3j,  0.8+1.6j]
            case 'random':
                np.random.seed(1926) # same values for every simulation run
                cc = np.random.normal(0, 1, size=channel_length) + 1j*np.random.normal(0, 1, size=channel_length)
            case 'custom':
                assert type(channel_coeffs) == np.ndarray
                cc = channel_coeffs
            case _:
                raise NotImplementedError(f"Channel type {channel_type} not supported!")
        self.lti_enabled = is_enabled
        self.channel_energy = np.sum(np.square(np.abs(cc)))
        self.channel_coeffs = cc / np.sqrt(self.channel_energy)