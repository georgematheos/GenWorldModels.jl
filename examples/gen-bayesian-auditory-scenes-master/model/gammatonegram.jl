using DSP;
using FFTW;

##another option: https://github.com/jfsantos/AuditoryFilters.jl

#Gammatonegram settings
function gtg_weights(sr, gtg_params)
    #=Generates a matrix of weights to combine FFT binds into Gammatone bins
    Input
    -----
    sr (samples/sec): sampling rate
    nfft: source fft size at sampling rate sr
    nfilts: number of output bands
    width (in Bark): constant width of each band
    minfreq, maxfreq (Hz): range covered
    maxlen: truncates number of bins to audible spectrum
    Returns
    -------
    wts: weight matrix
    cfreqs: actual centre frequencies of gammatone band in Hz
    =#

    twin=gtg_params["twin"]
    thop=gtg_params["thop"]
    minfreq=gtg_params["fmin"]
    maxfreq=sr/2
    nfft=DSP.Util.nextfastfft( Int(floor(2^(ceil(log(2*twin*sr)/log(2))))) )
    maxlen=Int(floor(nfft/2+1))
    nfilts=gtg_params["nfilts"]
    width=gtg_params["width"]
    
    #Constants, after Slaney's MakeERBFilters
    EarQ = 9.26449
    minBW = 24.7
    T = 1/sr
    nFr = 1:nfilts
    em = EarQ*minBW
    cfreqs = (maxfreq + em) * exp.(((-1*log(maxfreq + em) + log(minfreq + em))/nfilts)*nFr).-em
    cfreqs = cfreqs[end:-1:1]

    ERB = width*( cfreqs/EarQ .+ minBW )
    B = 1.019 * 2 * pi * ERB
    r = exp.(-B*T)
    cre = 2 * pi * cfreqs * T; cim = im*cre;

    ebt = exp.(B*T)
    ccpt = 2*T*cos.(cre); scpt = 2*T*sin.(cre)
    xm = sqrt(3 - 2^1.5); xp = sqrt(3 + 2^1.5)
    A(f::Function, x) = transpose(sr*0.5*( f.(ccpt./ebt, x*scpt./ebt) ))
    As = vcat(A(+, xp), A(-, xp), A(+, xm), A(-, xm))

    G(f::Function, x) = (-2*T*exp.(2*cim) .+ 2*T*exp.(-B.*T .+ cim).*( f.(cos.(cre), x*sin.(cre))))
    G_denom = ( -2*exp.(-2*B*T) - 2*exp.(2*cim) + 2*r.*(1 .+ exp.(2*cim)) ).^4
    gain = abs.(G(-, xm).*G(+, xm).*G(-, xp).*G(+, xp)./G_denom)

    uarr = (1:(nfft/2 + 1)) .- 1
    ucirc = repeat( transpose(exp.(2*im*pi*uarr/nfft)), outer=[nfilts,1])
    pole = repeat(r .* exp.(cim), outer=[1,size(ucirc)[2]])

    U(n) = abs.(ucirc .- repeat(As[n,:], outer=[1,size(ucirc)[2]]))
    Utg = (T^4)./repeat(gain, outer=[1, size(ucirc)[2]])
    Upole = abs.(((pole .- ucirc).*( conj.(pole) .- ucirc)).^(-4))

    wts = Utg.*U(1).*U(2).*U(3).*U(4).*Upole
    wts = wts[:,1:maxlen]

    return wts, cfreqs

end

function gammatonegram(x, wts, sr, gtg_params)
    #=Compute a fast cochleagram by weighting spectrogram channels with gammatone filters
    Input
    -----
    twin (sec): duration of windows for integration
    thop (sec): duration between successive integration windows
    nfilts: number of channels in gammatone filterbank
    fmin (Hz), fmax (Hz): range of frequencies covered
    width (relative to ERB default): how to scale bandwidths of filters
    Returns
    -------
    gammatonegram (in dB)
    vector of timepoints corresponding to gammatonegram bins
    center frequencies corresponding to gammatonegram bins (uniformly spaced on Bark scale)
    =#

    fmax = sr/2;
    #Define spectrogram settings
    nfft = DSP.Util.nextfastfft( Int(floor(2^(ceil(log(2*gtg_params["twin"]*sr)/log(2))))) )
    nhop = Int(round(gtg_params["thop"]*sr))
    nwin = Int(round(gtg_params["twin"]*sr))

    #Compute weights for channels of spectrogram
    if wts == []
        wts, cfreqs = gtg_weights(sr, gtg_params)
    end
    # I think this spectrogram might not be quite right
    gtg_window = DSP.hanning(nwin)
    sg = DSP.Periodograms.spectrogram(x, nwin, nwin-nhop, nfft=nfft, onesided=true, fs=sr, window=gtg_window)
    # multiply the power spectrogram by norm2*sr 
    # because that value is used to normalize the periodogram. MATLAB specgram allows you to
    # compute the magnitude spectrogram, and does not seem to include this normalization
    # Described here: https://github.com/JuliaDSP/DSP.jl/issues/98
    # And here: https://discourse.julialang.org/t/dsp-jl-power-spectra-normalization/26162
    # https://github.com/JuliaDSP/DSP.jl/blob/master/src/periodograms.jl#L497
    # where r is used as the normalization:
    # https://github.com/JuliaDSP/DSP.jl/blob/master/src/periodograms.jl#L66
    norm2=DSP.Periodograms.compute_window(gtg_window,nwin)[2]
    gtg = (1/nfft)*(wts*sqrt.(norm2*sr*DSP.power(sg)))
        
    ##Convert gammatonegram to dB
    gtg = gtg./gtg_params["ref"]
    gtg = max.(gtg, gtg_params["log_constant"])
    gtg = 20.0*log.(10, gtg) # use 20 because using amplitude for gtg
    gtg = max.(gtg, gtg_params["dB_threshold"])

    return gtg, time(sg)

end
