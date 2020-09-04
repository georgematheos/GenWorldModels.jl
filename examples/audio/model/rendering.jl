using Dierckx;
using Statistics: mean;
using Dierckx;
using FFTW;
using Random;
include("../model/time_helpers.jl"); #includes frequency conversions

#const lo_lim_freq = 1. in time_helpers
const rendering_log_const = 1e-12

## RAMPS
# Smooth onsets and offsets to reduce artifacts
function hann_ramp(x, sr)
    #= Applies a hann window to a soundwave s so s has gradual onsets & offsets
    x: soundwave
    sr (samples/sec)
    
    Parameters
    ----------
    ramp_duration (sec): duration of onset = duration of offset
    
    =#

    #Set parameters
    ramp_duration = 0.010
    
    #Make ramps
    t = 0:1/sr:ramp_duration
    n_samples = size(t)[1]
    off_ramp = 0.5*(ones(n_samples) + cos.( (pi/ramp_duration)*t ))
    on_ramp = off_ramp[end:-1:1]

    #Apply ramps
    x[1:n_samples] .*= on_ramp
    x[end-n_samples+1:end] .*= off_ramp

    return x

end

## WINDOWS 
# Creates windows in order to segment and differentially alter the excitation signal
function time_win_shape(x)
    return (1 .+ cos.(x))./2
end

function make_overlapping_time_windows(filt_length, tstep, sr)
    #=Computes array of linearly spaced cos-shaped windows 
    
    Input
    -----
    filt_length: length of time dimension of filter array
    tstep (s): duration of one filter element 
    sr (Hz): audio sampling rate
    
    Output
    ------
    win_arr (n_samples_in_scene, n_win): 
        overlapping cos-shaped windows for amplitude modulation in time
    
    =#
    win_len = 2*(floor(tstep*sr))
    sig_len = (filt_length - 1)*(win_len/2)
    
    # Check for erroneous inputs
    if (sig_len % (win_len/2) != 0) || (win_len % 2 != 0)
        error("win_len must be even and sig_len must be divisible by half of win_len")
    end
        
    n_win = Int(floor(2*sig_len/win_len + 1))
    win_adv_step = Int(floor(win_len/2))
    last_win_start_idx = Int(round(sig_len - floor(win_len/2)))
    
    window = time_win_shape(range(-pi, stop=pi, length=Int(win_len)))
    win_arr = zeros(Int(sig_len), n_win)
    win_idx_range = range(1, stop=last_win_start_idx, step=win_adv_step)
    for (curr_win_idx, curr_start_idx) in enumerate(win_idx_range)
        curr_lo_idx = curr_start_idx
        curr_hi_idx = curr_start_idx + Int(win_len) - 1
        win_arr[curr_lo_idx:curr_hi_idx, curr_win_idx + 1] = window
    end
    #Half windows at the edges
    n = Int(floor(win_len/2))
    win_arr[1:n, 1] = window[end-n+1:end]
    win_arr[end-n+1:end,end] = window[1:n]
    
    return win_arr, Int(sig_len)
    
end

function freq_win_shape(x)
    return cos.(x ./ 2.0)
end

function make_overlapping_freq_windows(filt_length, steps, audio_sr)
    #=Computes array of log spaced cos-shaped windows 
    
    Input
    -----
    filt_length: length of time dimension of filter array
    lowf (Hz), highf (Hz), estep (ERB): defines cutoffs, widths of frequency channels
    tstep (s): duration of one filter element 
    audio_sr (Hz): audio sampling rate
    
    Output
    ------
    win_arr (n_samples_in_scene, n_channels): 
        overlapping cos-shaped windows for amplitude modulation in frequency
    
    =#
    
    n_freqs = Int((filt_length - 1)*floor(steps["t"]*audio_sr))
    max_freq = audio_sr/2. - 1; 
    
    if steps["scale"] == "ERB" 
        freq_to_scale = freq_to_ERB 
        scale_to_freq = ERB_to_freq
    elseif steps["scale"] == "octave"
        freq_to_scale = freq_to_octave
        scale_to_freq = octave_to_freq
    end
    
    loscale = freq_to_scale(lo_lim_freq)
    hiscale = freq_to_scale(max_freq)
    n_channels = Int(length(get_element_gp_freqs(audio_sr, steps)) + 2);
    freqs = range(0, stop=max_freq, length=n_freqs)    

    scale_cutoffs_1D = range(loscale, stop=hiscale, length=n_channels)
    scale_cutoffs = [[scale_cutoffs_1D[i], scale_cutoffs_1D[i+2]] for i in range(1,stop=n_channels - 2)] #50% overlap
    freq_cutoffs = [scale_to_freq(s) for s in scale_cutoffs]
    
    win_arr = zeros(n_freqs, n_channels)
    for curr_channel = range(1,stop=n_channels - 2)
        
        curr_lo_freq = freq_cutoffs[curr_channel][1]
        curr_hi_freq = freq_cutoffs[curr_channel][2]
        curr_lo_freq_idx = argmax(freqs .> curr_lo_freq)
        curr_hi_freq_idx = argmin(freqs .< curr_hi_freq)

        curr_lo_scale = scale_cutoffs[curr_channel][1]
        curr_hi_scale = scale_cutoffs[curr_channel][2]
        curr_mean_scale = (curr_hi_scale + curr_lo_scale)/2
        scale_bandwidth = curr_hi_scale - curr_lo_scale

        curr_scale = freq_to_scale(freqs[curr_lo_freq_idx:curr_hi_freq_idx-1])
        normalized_domain = 2*(curr_scale .- curr_mean_scale)/scale_bandwidth
        curr_win = freq_win_shape(pi.*normalized_domain)
        win_arr[curr_lo_freq_idx:curr_hi_freq_idx-1, curr_channel + 1] = curr_win
        
    end
    
    return win_arr[:,2:end-1], freq_cutoffs
    
end
 
## EXCITATION
function FM_excitation(erbf0, sig_len, tstep, sr)
    #= Generate a frequency modulated tone 
    
    Input
    -----
    erbf0 (vector of ERB elements): specification of frequency in each window
    sig_len (number of samples): overall duration of window vec in samples
    tstep (s): duration of one time window in erbf0 vector
    sr (Hz): audio sampling rate
    
    Returns
    -------
    1D vector of frequency modulated tone
    
    =#
    
    timepoints = range(0, stop=(sig_len - 1)/sr, length=Int(round(sig_len)))
    erbSpl = Spline1D(tstep.*(0:length(erbf0)-1), erbf0, k=1)
    unclamped_f0 = ERB_to_freq(erbSpl(timepoints))
    f0 = clamp.(unclamped_f0, lo_lim_freq, sr/2. - 1)
    
    source = sin.(2*pi*cumsum(f0*(1/sr)))
    source /= (mean(source.^2)).^0.5
    
    return source, unclamped_f0, f0
end

function white_excitation(sig_len, sr) #duration, sr)
    #= Generate bandpass white noise 
    Input
    -----
    duration (s)
    sr (sampling rate, Hz)
    
    Returns
    -------
    1D vector of constant amplitude white noise
    =#
    
    
    nyq_freq = Int(floor(sr/2))
    hi_lim_freq = nyq_freq - 1
    #sig_len = Int(floor(sr*duration))
    #fft_len = DSP.Util.nextfastfft(sig_len)
    lo_idx = Int(ceil(lo_lim_freq/(1. * nyq_freq)*sig_len))
    hi_idx = Int(floor(hi_lim_freq/(1. * nyq_freq)*sig_len))

    noise_spec = zeros(sig_len)
    ##Using the noise_rng is necessary for making the trace and round_trip_trace have the same score
    noise_rng = MersenneTwister(1234); ## SHOULD THIS BE A RANDOM VARIABLE with its own MCMC move? 
    noise_spec[lo_idx:hi_idx-1] = randn(noise_rng, Float64, (hi_idx - lo_idx,)) #randn((hi_idx - lo_idx,))
    source = FFTW.idct(noise_spec)
    source /= (mean(source.^2)).^0.5
    return source

end

function pink_excitation(sig_len, sr) #duration, sr)
    #= Generate bandpass white noise 
    Input
    -----
    duration (s)
    sr (sampling rate, Hz)
    
    Returns
    -------
    1D vector of constant amplitude white noise
    =#
    
    
    nyq_freq = Int(floor(sr/2))
    hi_lim_freq = nyq_freq - 1
    #sig_len = Int(floor(sr*duration))
    #fft_len = DSP.Util.nextfastfft(sig_len) --> nextprod
    lo_idx = Int(ceil(lo_lim_freq/(1. * nyq_freq)*sig_len))
    hi_idx = Int(floor(hi_lim_freq/(1. * nyq_freq)*sig_len))
    binFactor = float(sig_len/sr)
    
    x_real = zeros(sig_len)
    x_imag = zeros(sig_len)
    noise_rng_real = MersenneTwister(1234);
    noise_rng_imag = MersenneTwister(2134);
    x_real[lo_idx:hi_idx-1]=randn(noise_rng_real, Float64, (hi_idx-lo_idx,))
    x_imag[lo_idx:hi_idx-1]=randn(noise_rng_imag, Float64, (hi_idx-lo_idx,))
    spectrum = x_real + x_imag*1im
    
    # divide each element of the spectrum by f^(1/2).
    # Since power is the square of amplitude, 
    # this means we divide the power at each component by f
    # if you want to include arbitrary beta parameter, use instead power(pinkWeights,beta/2.)
    # beta = 0: white noise,  beta = 2: red (Brownian) noise
    pinkWeights = binFactor*range(1,stop=sig_len)
    pinkSpectrum = spectrum ./ sqrt.(pinkWeights)
    
    noise = abs.(FFTW.idct(pinkSpectrum))
    noise = noise[1:sig_len]
    noise /= sqrt(mean(noise.^2.))

    return noise
    
end

function harmonic_excitation(erbf0, n_harmonics, sig_len, steps, audio_sr)
    
    ##Generating harmonic stack source with frequency modulation
    hi_lim_freq = audio_sr/2. - 1;
    hi_lim_ERB = freq_to_ERB(floor(hi_lim_freq))
    
    #Sum up the FM harmonics
    f0 = ERB_to_freq(erbf0)
    excitation = zeros(Int(round(sig_len)))
    clamped_f0=[]
    for i = 1:n_harmonics
        f = f0 .* i
        if all(f .> hi_lim_freq)
            break
        end
        f = [f_ < lo_lim_freq ? lo_lim_freq : f_ for f_ in f]
        erbf = freq_to_ERB(f)
        h, unclamped_f, clamped_f = FM_excitation(erbf, sig_len, steps["t"], audio_sr)
        h = [ unclamped_f[j] > hi_lim_freq ? 0.0 : h[j] for j in 1:length(h)]
        excitation += h
        if i == 1
            clamped_f0 = clamped_f
        end
    end
    
    #Don't make the excitation pink here:
    #It should be pink wrt to the fundamental frequency
    #That way, if a sound gets higher in frequency, it doesn't
    #automatically also get quieter!
#     fft_sig = FFTW.dct(excitation)
#     binFactor = float(sig_len/audio_sr)
#     pinkWeights = binFactor*range(1,stop=sig_len)
#     fft_sig = fft_sig ./ sqrt.(pinkWeights)
#     excitation = FFTW.idct(fft_sig)
    
    excitation /= sqrt(mean(excitation.^2.)) ##Should this be here?
    
    return excitation, clamped_f0
    
end

## SUBBANDS
function generate_subbands(x, filterbank)
    #=Split a sound into several frequency subbands
    
    Input
    -----
    x: 1D signal
    filterbank (length(x), n_freq_channels): log-spaced cosine filters 
    =#
    
    sig_len = length(x)
    filt_len, n_channels = size(filterbank)
    if sig_len != filt_len
        error("Signal length ($sig_len) must equal filter length ($filt_len).")
    end
    
    #fft_len = DSP.Util.nextfastfft(sig_len)
    fft_sig = FFTW.dct(x)
    filtered_subbands = filterbank .* repeat(fft_sig, 1, n_channels)
    subbands = FFTW.idct(filtered_subbands, 1)
    return subbands 
    
end

function collapse_subbands(subbands, filterbank)
    #=Combine subbands into a single sound
    
    Input
    -----
    subbands (length(x), n_channels): subbands from "generate_subbands"
    filterbank (length(x), n_channels): log-spaced cosine filters 
    
    =#
    
    sig_len = size(subbands)[1]
    filt_len = size(filterbank)[1]
    if sig_len != filt_len
        error("Signal length must equal filter length.")
    end
    #fft_len = DSP.Util.nextfastfft(sig_len)
    fft_subbands = FFTW.dct(subbands, 1)
    filtered_subbands = filterbank.*fft_subbands
    mod_subbands = FFTW.idct(filtered_subbands, 1)
    return sum(mod_subbands, dims=2)
    
end

function modulate_subbands(subbands, win_arr, energy_grid, tstep, sr)
    #=Change the amplitude of subbands in each time element & frequency subband    
    
    Input
    -----
    subbands (length(x), n_channels): subbands from "generate_subbands"
    win_arr (length(x), n_windows): linearly-spaced cosine filters in time
    energy_grid (n_channels, n_windows): log-spaced cosine filters in frequency
    
    Returns
    -------
    modified subbands (length(x), n_channels)
    
    =#
    
    clen, n_channels = size(subbands)
    wlen, n_windows = size(win_arr)
    if clen == wlen
        sig_len = clen
    else
        error("Incorrect array lengths.")
    end
    nonzero(a) = a != 0.
    
    mod_subbands = zeros(size(subbands))
    for curr_sub_i = 1:n_channels
        curr_subband = subbands[:, curr_sub_i]
        mod_subband_accum = zeros(size(curr_subband))
        for curr_win_i = 1:n_windows
            curr_win = win_arr[:,curr_win_i] 
            indices = findall(nonzero, curr_win)
            windowed_subband = curr_subband.*curr_win
            noise_rms = sqrt(mean((windowed_subband[indices]).^2.)) ##??
            mod_subband_accum += energy_grid[curr_sub_i,curr_win_i] .*windowed_subband./noise_rms
        end
        mod_subbands[:, curr_sub_i] = mod_subband_accum
    end
    
   return mod_subbands
end    

## ELEMENT GENERATION
function generate_tone(erbf0, filt, duration, tstep, sr, rms_ref)
    #= Generate an amplitude modulated, frequency modulated tone
    
    Input
    -----
    erbf0: 1D vector of ERB values 
    filt: 1D vector of decibel values
    duration (s): overall duration of sound
    tstep (s): duration of one element of erbf0 or filt
    ---> Right now, erbf0 and filt have the same tstep, 
         but it could be different depending on what we 
         want the GP sampling rates to be. 
    sr (Hz): audio sampling rate
    
    Returns
    -------
    1D vector of am/fm tone with ramps
    =#
    
    #Create filter
    #filt = filt_scale.(filt)
    erbf0 = vcat([erbf0[1]], erbf0, [erbf0[end]])
    filt = vcat([filt[1]], filt, [filt[end]])
    win, sig_len = make_overlapping_time_windows(length(erbf0), tstep, sr)
    energy_grid = rms_ref * 10.0 .^(filt/20.) .- rendering_log_const
    
    #Create excitation
    FM_tone, unclamped_f0, clamped_f = FM_excitation(erbf0, sig_len, tstep, sr)
    
    #Apply filter
    A = win .* reshape(energy_grid, 1, length(energy_grid)) 
    AM_FM_tone = sum(FM_tone .* A,dims=2)
    
    n_samples = Int(floor(sr*duration))
    start_point = max(1, Int(floor((length(AM_FM_tone) - n_samples)/2)))
    AM_FM_tone = AM_FM_tone[start_point:start_point+n_samples-1]
    
    tone = hann_ramp(AM_FM_tone[:,1],sr)
    
    return tone

end

function modulate_noise(subbands, corr_grid, win_arr, filterbank, steps, audio_sr, rms_ref, freq_cutoffs)
    #=modulate noise with gp-sampled amplitudes for each (t,f) bin
    
    Input
    -----
    subbands (length(x), n_channels):
    corr_grid (n_windows, n_channels):
    win_arr (length(x), n_windows):
    filterbank (length(x), n_channels):
    
    Returns
    -------
    1D vector of amplitude modulated noise
    =#

    n_channels = size(subbands)[2]
    n_windows = size(win_arr)[2]

    #Hartmann page 49: spectrum_level = sound pressure level - 10 log (BW)
    # sound_pressure_level = spectrum_level + 10log(BW)
    bandwidthsHz = repeat(reshape([Hz[2] - Hz[1] for Hz in freq_cutoffs[end:-1:1]], 1, n_channels),n_windows,1) #inversion necessary to line up with the corr_grid, to make sure noise comes out appropriately (eg pink spectrum looks flat on cochleagram) 
    #corr_grid = spectrum_level so add correction
    energy_grid = transpose( rms_ref*10.0 .^( (corr_grid + 10*log.(10, bandwidthsHz))/20.0 ) .- rendering_log_const)
    mod_subbands = modulate_subbands(subbands, win_arr, energy_grid, steps["t"], audio_sr)
    noise = collapse_subbands(mod_subbands, filterbank)

    return noise[:,1]
    
end

function generate_noise(filt, duration, steps, audio_sr, rms_ref)
    #=Generate time-varying noise
    
    Input
    -----
    duration (s): overall duration of noise segment
    filt (n_windows, n_channels): 2D GP sampled amplitudes
    lowf (Hz), highf (Hz), estep (ERB): defines cutoffs, widths of frequency channels
    tstep (s): duration of one filter element 
    sr (Hz): audio sampling rate
    
    Returns
    -------
    1D vector of amplitude modulated noise
    
    =#

    #Create filter
    #filt = filt_scale.(filt)
    
    #Time & freq windows for amplitude modulation
    filt = vcat( reshape(filt[1,:],(1,size(filt)[2])), filt, reshape(filt[end,:],(1,size(filt)[2])))
    win_arr, sig_len = make_overlapping_time_windows(size(filt)[1], steps["t"], audio_sr) 
    filterbank, freq_cutoffs = make_overlapping_freq_windows(size(filt)[1], steps, audio_sr)
    
    #Source & subbands
    colour = "pink" #or "white 
    source = colour == "white" ? white_excitation(sig_len, audio_sr) : pink_excitation(sig_len, audio_sr)
    subbands = generate_subbands(source, filterbank)

    #Amplitude modulate noise 
    AM_noise = modulate_noise(subbands, filt, win_arr, filterbank, steps, audio_sr, rms_ref, freq_cutoffs)
    
    n_samples = Int(floor(audio_sr*duration))
    start_point = max(1, Int(floor((length(AM_noise) - n_samples)/2)))
    AM_noise = AM_noise[start_point:start_point+n_samples-1]
    
    #Apply ramps
    noise = hann_ramp(AM_noise[:,1],audio_sr)
    
    return noise
    
end

function modulate_harmonic(excitation, filt, f0, win_arr, filterbank, freq_cutoffs, rms_ref)
    
    wlen, n_windows = size(win_arr)
    flen, n_channels = size(filterbank)
    if wlen == flen
        sig_len = wlen
    else
        error("Incorrect array lengths")
    end
    #Convert spectrum level into energy
    bandwidthsHz = repeat(reshape([Hz[2] - Hz[1] for Hz in freq_cutoffs[end:-1:1]], 1, n_channels),n_windows,1) 
    energy_grid = transpose( rms_ref*10.0 .^( (filt + 10*log.(10, bandwidthsHz))/20.0 ) .- rendering_log_const)

    #amplitude modulation 
    AM_FM_tone = zeros(Int(round(sig_len)))
    nonzero(a) = a != 0.
    for curr_win_i = 1:n_windows
        
        #section of source to modulate
        curr_win = win_arr[:,curr_win_i] 
        indices = findall(nonzero, curr_win)
        excitation_i = excitation.*curr_win
        harmonic_rms = sqrt(mean((excitation_i[indices]).^2.)) ##??        
        norm_excitation_i = excitation_i ./ harmonic_rms
        
        #preparing filter
        energy_i = energy_grid[:, curr_win_i] #n_channels
        energy_i = reshape(energy_i, (1, n_channels))
        energy_i = repeat(energy_i, sig_len, 1)
        weighted_filterbank = energy_i .* filterbank
        filter = sum(weighted_filterbank, dims=2)
        
        #applying filter
        fft_sig = FFTW.dct(norm_excitation_i)
        #Make it pink!
#         mean_f0 = mean(f0[indices])
#         freqs = float.(range(0, stop=9999, length=length(norm_excitation_i)))
#         mean_f0_idx = argmin(abs.(freqs .- mean_f0))
#         pinkWeights = float.(range(1, stop=length(norm_excitation_i)))
#         pinkWeights[1:mean_f0_idx] .= 0 
#         pinkWeights /=  maximum(pinkWeights)
#         pinkWeights *= float(length(indices)/audio_sr)
#         fft_sig[mean_f0_idx+1:end] = fft_sig[mean_f0_idx+1:end] ./ sqrt.(pinkWeights[mean_f0_idx+1:end])
#         plot(fft_sig); 
        #Apply additional filter
        filtered_excitation_i = FFTW.idct(filter .* fft_sig )
        AM_FM_tone += filtered_excitation_i
        
    end
    
    return AM_FM_tone
    
end

function generate_harmonic(erbf0, filt, duration, n_harmonics, steps, audio_sr, rms_ref)
    
    ##Preparing windows & filters; prepare latents
    #Latent variables resize for edges of element
    erbf0 = vcat([erbf0[1]], erbf0, [erbf0[end]])
    filt = vcat( reshape(filt[1,:],(1,size(filt)[2])), filt, reshape(filt[end,:],(1,size(filt)[2])))
    #Generate window & filter
    win_arr, sig_len = make_overlapping_time_windows(size(filt)[1], steps["t"], audio_sr) 
    filterbank, freq_cutoffs = make_overlapping_freq_windows(size(filt)[1], steps, audio_sr) 

    excitation, f0 = harmonic_excitation(erbf0, n_harmonics, sig_len, steps, audio_sr)
    AM_FM_tone = modulate_harmonic(excitation, filt, f0, win_arr, filterbank, freq_cutoffs, rms_ref) 
        
    n_samples = Int(floor(audio_sr*duration))
    start_point = max(1, Int(floor((length(AM_FM_tone) - n_samples)/2)))
    AM_FM_tone = AM_FM_tone[start_point:start_point+n_samples-1]
    
    #Apply ramps
    AM_FM_tone = hann_ramp(AM_FM_tone[:,1],audio_sr)
    
    return AM_FM_tone
    
end
