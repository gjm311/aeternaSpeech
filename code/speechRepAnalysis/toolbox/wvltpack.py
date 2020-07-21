import numpy as np
import numpy.fft



class wvltPack:
    
    def __init__(self, signal, dil=1.1, nbf=16):
        
        self.signal=signal
        self.fs=np.shape(signal)[0]
        self.dil=dil
        self.nbf=nbf
        self.phi=None
        self.psi=None
        
        
    def get_wvlts(self):
        FS=16000
        FRAME_SIZE=0.5
        TIME_SHIFT=0.25
        NFR=16
        OVRLP=0.75
        FRM_SZ=int(FRAME_SIZE*FS/NFR)
        TS=int(FRM_SZ*OVRLP)
        
        init=0
        endi=int(FS*FRAME_SIZE)
        nf=int(len(self.signal)/(TIME_SHIFT*FS))-1
        
        self.create_wavelets(FRM_SZ)
        fpsis=np.zeros((nf,NFR,self.nbf,FRM_SZ))
      
        for k in range(nf):
#             try:
                frame=self.signal[init:endi]                  
                init=init+int(TIME_SHIFT*FS)
                endi=endi+int(TIME_SHIFT*FS)
                
                inin=0
                enin=FRM_SZ
                
                for kk in range(NFR):
                    framein=frame[inin:enin]                       
                    inin=inin+int(TS)
                    enin=enin+int(TS)
                    if len(framein)<FRM_SZ:
                        framein=np.pad(framein,FRM_SZ-len(framein))
                        framein=framein[:FRM_SZ]

                    fpsis[k,kk,:,:],fphi=self.wavelet_transform(framein)

#                 fpsi,fphi=wavelet_transform(frame)
#                 fpw=np.mean(fpsi**2,axis=1)
#                 wv_mat[:,k,:]=fpw
#             except:
#                 init=init+TIME_SHIFT
#                 endi=endi+TIME_SHIFT

        return fpsis

        
    def create_wavelets(self,fr_size,volta=0):
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

        freqs=[(fr_size/2)*np.power(1/self.dil,bf) for bf in np.arange(self.nbf)]

        ms=np.tile(freqs,(fr_size,1)).T
        oms=np.tile(np.arange(fr_size),(self.nbf,1))

        p=0.3/(np.log(self.dil)/np.log(2))**2
        c=np.exp(-p) 

        d=1
        for k in range(10):
            d=1/(1-np.exp(-p*2*d))

        exp1=-p*(np.divide(d*oms,(ms-1))**2)
        exp2=-p*(np.divide(d*oms,ms)**2)

        psi=np.exp(exp1)-c*np.exp(exp2)

        max_psi=np.max(abs(psi))
        self.psi=np.float32(np.divide(psi,max_psi))
        self.phi=np.exp(-np.arange(fr_size)**2)
        self.freqs=freqs
        
        if volta==1:
            return self.freqs,self.psi,self.phi


    def wavelet_transform(self,signal,psi=None,phi=None):
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

        if psi==None and phi==None:
            phi=self.phi
            psi=self.psi
        
        signal=signal.T
        if len(signal) != np.shape(psi)[1]:
            print('The sizes of the wavelets, '+str(np.shape(psi))+' and the signal '+str(len(signal))+' do not match!')
                  
        ft_sig=np.fft.fft(signal)

        fpsi=np.fft.ifft(np.multiply(psi,ft_sig))
        fphi=np.fft.ifft(np.multiply(ft_sig,phi))
      
        return fpsi,fphi
