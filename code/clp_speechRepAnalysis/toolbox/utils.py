import numpy as np
import numpy.fft



def create_wavelets(fr_size,nbf=32,dil=1.25):
    """
    Compute wavelet basis function for 'morl' wavelets
    ADAPTED from Phase retrieval for wavelet transforms: 
    http://www-math.mit.edu/~waldspur//wavelets_phase_retrieval.html#source_code

    :param fs: number of samples (should be equivalent to signal that will be transformed)
    :param dil: dilation factor between wavelets.
    :returns freqs: frequencies related to wavelets.
    :returns psi: family of wavelets with each wavelet being a row.
    :returns phi: gaussian low freq. function.
    """

    old_settings = np.seterr(under='ignore')

    freqs=[(fr_size/2)*np.power(1/dil,bf) for bf in np.arange(nbf)]

    ms=np.tile(freqs,(fr_size,1)).T
    oms=np.tile(np.arange(fr_size),(nbf,1))

    p=0.3/(np.log(dil)/np.log(2))**2
    c=np.exp(-p) 

    d=1
    for k in range(10):
        d=1/(1-np.exp(-p*2*d))

    exp1=-p*(np.divide(d*oms,(ms-1))**2)
    exp2=-p*(np.divide(d*oms,ms)**2)

    psi=np.exp(exp1)-c*np.exp(exp2)

    max_psi=np.max(abs(psi))
    psi=np.float32(np.divide(psi,max_psi))
    phi=np.exp(-np.arange(fr_size)**2)
    freqs=freqs

    return freqs,psi,phi


def wavelet_transform(signal,psi,phi=None):
    """
    Compute wavelet transform of a signal
    ADAPTED from Phase retrieval for wavelet transforms: 
    http://www-math.mit.edu/~waldspur//wavelets_phase_retrieval.html#source_code

    :param signal: signal to be transformed
    :param psi: family of wavelets with each wavelet being a row.
    :param phi: gaussian low freq. function.
    :returns fpsi: components of the wavelet transform (time domain rep.).
    :returns fphi: conv. of signal with low-freq. filter
    """
    
    signal=signal.T
    if len(signal) != np.shape(psi)[1]:
        print('The sizes of the wavelets, '+str(np.shape(psi))+' and the signal '+str(len(signal))+' do not match!')

    ft_sig=np.fft.fft(signal)

    fpsi=np.fft.ifft(np.multiply(psi,ft_sig))
    fphi=np.fft.ifft(np.multiply(ft_sig,phi))

    return fpsi,fphi
