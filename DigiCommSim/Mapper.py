#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

class Mapper:
    """
    A class used to map bits and symbols for different modulation schemes, and generate bitstreams.

    Attributes
    ----------
    mod_scheme : str
        The modulation scheme

    Methods
    -------
    generate_bits
        Generates a bitstream for given length and type
    map_bits
        Maps bit to symbols
    demap_symbols
        Maps symbols to bits
    """
    
    def __init__(self, mod_scheme: str='bpsk'):
        """
        Parameters
        ----------
            mod_scheme : str, optional
                The modulation scheme (default is 'bpsk')
        Raises
        ------
            NotImplementedError
                If the given modulation scheme is not supported.
        """
        if mod_scheme not in  ['bpsk', 'qpsk', '16qam']:
            raise NotImplementedError(f"Modulation scheme {mod_scheme} not implemented!")
        self.mod_scheme = mod_scheme

    def generate_bits(self, n_bits: int, bit_gen_mode: str='random', seed: int=1926) -> np.ndarray:
        """
        Generates random bitstream.

        Parameters
        ---------
            n_bits : float
                The number of bits to generate
            bit_gen_mode: str, optional
                The type of bits to generate (default  is 'random')
            seed: int, optional
                The seed for numpy RNG module (default is '1926')
        
        Raises
        ------
            NotImplementedError
                If the given bit generation mode is not supported.
        
        Returns
        -------
            ndarray
                The generated bitstream
        """
        match bit_gen_mode:
            case 'zeros':
                return np.zeros(n_bits)
            case 'ones':
                return np.ones(n_bits)
            case 'random':
                np.random.seed(seed)
                return np.random.randint(2, size=n_bits)
            case _:
                raise NotImplementedError(f"Bit generation mode {bit_gen_mode} not supported!")

    def map_bits(self, bit_vector: np.ndarray) -> np.ndarray:
        """
        Maps bits to symbols.
        
        Parameters
        ---------
            bit_vector : ndarray
                The original bit vector
            
        Returns
        -------
            ndarray
                The mapped symbol vector
        """
        method_name = f"_mapper_{self.mod_scheme}"
        return getattr(self, method_name)(bit_vector)

    def _mapper_bpsk(self, bit_vector: np.ndarray) -> np.ndarray:
        """
        Maps symbols according to the BPSK constellation diagram. 
        
        Parameters
        ---------
            bit_vector : ndarray
                The original bit vector
            
        Returns
        -------
            ndarray
                The mapped symbol vector
        """
        return bit_vector * (-2) + 1

    def _mapper_qpsk(self, bit_vector: np.ndarray) -> np.ndarray:
        """
        Map symbols according to table 7.1.2-1.
        
        Parameters
        ---------
            bit_vector : ndarray
                The original bit vector
          
        Raises
        ------
            ValueError
                If the given bit vector can not be mapped due to invalid length.
        
        Returns
        -------
            ndarray
                The mapped symbol vector
        """
        if len(bit_vector) % 2 != 0:
            raise ValueError(f"Can not map bit_vector of length {len(bit_vector)} to QPSK symbols!")
        i_amp = bit_vector[::2] * (-2) + 1
        q_amp = bit_vector[1::2] * (-2) + 1
        return np.array(1 / np.sqrt(2) * (i_amp + 1j*q_amp), dtype=complex)

    def _mapper_16qam(self, bit_vector: np.ndarray) -> np.ndarray:
        """
        Map symbols according to table 7.1.3-1.
        
        Parameters
        ---------
            bit_vector : ndarray
                The original bit vector
          
        Raises
        ------
            ValueError
                If the given bit vector can not be mapped due to invalid length.
        
        Returns
        -------
            ndarray
                The mapped symbol vector
        """
        if len(bit_vector) % 4 != 0:
            raise ValueError(f"Can not map bit_vector of length {len(bit_vector)} to 16QAM symbols!")
        i_sign = bit_vector[::4] * (-2) + 1
        q_sign = bit_vector[1::4] * (-2) + 1
        i_amp = bit_vector[2::4] * 2 + 1
        q_amp = bit_vector[3::4] * 2 + 1
        return np.array(1/np.sqrt(10) * (i_sign * i_amp + 1j*q_sign * q_amp), dtype=complex)

    def demap_symbols(self, symbol_vector: np.ndarray) -> np.ndarray:
        """
        Maps symbols to bits.
        
        Parameters
        ---------
            symbol_vector : ndarray
                The original symbol vector
            
        Returns
        -------
            ndarray
                The demapped bit vector
        """
        method_name = f"_demapper_{self.mod_scheme}"
        return getattr(self, method_name)(symbol_vector)

    def _demapper_bpsk(self, symbol_vector: np.ndarray) -> np.ndarray:
        """
        Demap symbols according to the BPSK constellation diagram. 
        
        Parameters
        ---------
            symbol_vector : ndarray
                The original symbol vector
        
        Returns
        -------
            ndarray
                The demapped bit vector
        """
        return np.where(symbol_vector >= 0, 0, 1)

    def _demapper_qpsk(self, symbol_vector: np.ndarray) -> np.ndarray:
        """
        Demap symbols according to table 7.1.2-1.
        
        Parameters
        ---------
            symbol_vector : ndarray
                The original symbol vector
        
        Returns
        -------
            ndarray
                The demapped bit vector
        """
        bit_vector = np.zeros(len(symbol_vector)*2, dtype=int)
        bit_vector[::2] = np.where(np.real(symbol_vector) >= 0.0, 0, 1)
        bit_vector[1::2] = np.where(np.imag(symbol_vector) >= 0.0, 0, 1)
        return bit_vector

    def _demapper_16qam(self, symbol_vector: np.ndarray) -> np.ndarray:
        """
        Demap symbols according to table 7.1.3-1.
        
        Parameters
        ---------
            symbol_vector : ndarray
                The original symbol vector
        
        Returns
        -------
            ndarray
                The demapped bit vector
        """
        bit_vector = np.zeros(len(symbol_vector)*4, dtype=int)
        bit_vector[::4] = np.where(np.real(symbol_vector) >= 0.0, 0, 1)
        bit_vector[1::4] = np.where(np.imag(symbol_vector) >= 0.0, 0, 1)
        bit_vector[2::4] = np.where(np.absolute(np.real(symbol_vector)) >= 2/np.sqrt(10), 1, 0)
        bit_vector[3::4] = np.where(np.absolute(np.imag(symbol_vector)) >= 2/np.sqrt(10), 1, 0)
        return bit_vector