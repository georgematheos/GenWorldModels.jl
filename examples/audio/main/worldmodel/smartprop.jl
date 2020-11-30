using DSP: conv
include("../folded_normal.jl")
# get audiograms from traces
underlying_gram(tr) = AudioInference.get_retval(tr)[1]
observed_gram(tr) = tr[:kernel => :scene]
error_gram(tr) = observed_gram(tr) - underlying_gram(tr)

# threshold an image by the given value as a fraction of the max val in the image
function get_thresholded(img, threshold)
    threshold = threshold * maximum(img)
    if threshold < 0
        return img .< threshold
    else
        return img .> threshold
    end
end

# normalize an image so the max value is 1 and min is 0
normalize(img) = 1/(maximum(img) - minimum(img)) * img

# given a strip (row vector), estimates where there are "changepoints" from
# a region of one intensity to another
function get_changepoints(strip; threshold_frac=0.2)
    # TODO: I could possibly be smarter by not doing abs, and looking at start vs end of regions
    cnvd = @view abs.(conv(strip/maximum(strip), [-2 -1 0 1 2]))[2:length(strip)+1]
    changepts = []
    threshold = maximum(cnvd) * threshold_frac
    previdx = 0
    for (i, val) in enumerate(cnvd)
        if val > threshold
            if previdx < i - 1
                push!(changepts, i)
            end
            previdx = i
        end
    end
    return changepts
end

# given a column bitvector, returns the connected regions
# format: [(a, b), (c, d), (e, f), ...]
function get_regions(col)
    regions = []
    start = nothing
    last_was_true = false
    for i=1:length(col)
        if col[i] > 0
            if !last_was_true
                start = i
            end
            last_was_true = true
        else
            if last_was_true
                push!(regions, (start, i-1))
                start = nothing
            end
            last_was_true = false
        end
    end
    if last_was_true
      push!(regions, (start, length(col)))
    end
    return regions
end

# given a column from one horizontal region of the soundwave,
# this approximately determines whether it's a noise or a tone,
# and the volume if it is a noise / freq if it is a tone
function analyze_col(col, width)
    thresholded = get_thresholded(col, 0.01)
    regions = get_regions(thresholded)
    is_noise = length(regions) > 0 && sum(nd-st for (st, nd) in regions) > length(thresholded)/0.7
    if is_noise
        analyze_noise(col, width)
    else
        analyze_tones(col, regions)
    end
end

# I derived these formulae by analyzing the generated
# noises/tones at different amp/erb values:
noise_amp_for_col(col) = 1054 + 64*sum(col)
noisesum_for_amp(amp) = (amp - 1054)/(64)
function pos_for_erb_val(x)
    if x ≤ 5
        9 + 2x
    elseif x ≤ 12
        19 + (31/7)*(x-5)
    elseif x ≤ 15
        50 + 3(x-12)
    else
        60.5
    end
end
function tone_erb_for_val(x)
    if x ≤ 19
        0.5(x-9)
    elseif x ≤ 50
        5. + (7.0/31.0)*(x-10.)
    elseif x ≤ 60
        12. + (1.0/3.0)*(x-50.)
    else
        15.
    end
end

# returns `(:noise, amp)` where `amp` is roughly the amplitude of
# the noise in this column
function analyze_noise(col, width)
    return (:noise, noise_amp_for_col(col/width))
end
# returns `(:tones, vec)` where `vec` is a vector of approximate
# erb values for which there is a tone in this column
function analyze_tones(col, regions)
    return (:tones, [tone_erb_for_val((x+y)/2) for (x,y) in regions])
end

### get_analyzed_img_sections ###
function get_analyzed_audiogram_sections(gram)
    changepoints = get_changepoints(normalize(sum(gram, dims=1)))
    noises = []
    tones = []
    for i=1:length(changepoints)-1
        region = (changepoints[i], changepoints[i+1])
        col = sum(@view(gram[:, region[1]:region[2]]), dims=2)
        (typ, params) = analyze_col(col, region[2] - region[1])
        if typ === :noise
            push!(noises, (region, params))
        else
            for mean in params
                push!(tones, (region, mean))
            end
        end
    end
    return (noises, tones)
end
function analyze_gram(gram)
    pos = map(x -> x > 0 ? x : 0, gram)
    # neg = map(x -> x < 0 ? x : 0, err)
    get_analyzed_audiogram_sections(pos)
end
function analyze_errorgram(tr)
    analyze_gram(error_gram(tr))
end

function score_possibilities(possibilities, λ=1.)
    TONE_COL_SUM = 526.
    function impact(possibility)
        region = possibility[2][1]
        len = region[2] - region[1]
        if possibility[1] === :noise
            len * noisesum_for_amp(possibility[2][2])
        else
            len * TONE_COL_SUM
        end
    end

    # not 100% sure this will work:
    prescores = @. λ * impact(possibilities)

    sm = logsumexp(prescores)
    scores = @. exp(prescores - sm)

    return scores
end

const STARTTIME_STD = 0.05
const DURATION_STD = 0.05
const AMP_STD = 4.0
const ERB_STD = 1.

# length of overlap between 2 regions
function overlap((a, b), (c, d))
    if a > c
        return overlap((c, d), (a, b))
    end
    if b < c
        return 0
    end
    if d < b
        return (d - c)
    end
    return c - b
end

# ER_AFTER_DEL = nothing
# ANALYSIS = nothing
@dist list_categorical(list, probs) = list[categorical(probs)]
@gen function smart_bd_proposal(tr)
    birthprior = tr[:kernel => :n_tones] == 4 ? 0. : 0.5
    do_birth ~ bernoulli(birthprior)
    if do_birth
        birth_idx ~ uniform_discrete(1, tr[:kernel => :n_tones] + 1)

        (noises_to_add, tones_to_add) = analyze_errorgram(tr)

        # if ER_AFTER_DEL !== nothing
        #     if !isapprox(error_gram(tr), ER_AFTER_DEL)
        #         display((error_gram(tr) - ER_AFTER_DEL)[30:50, 100:120])
        #     end
        #     println("ISAPPROX: ", isapprox(error_gram(tr), ER_AFTER_DEL), "; isequal: ", (error_gram(tr) == ER_AFTER_DEL))
        #     if ANALYSIS != (noises_to_add, tones_to_add)
        #         println("death got ", ANALYSIS, "; birth got ", (noises_to_add, tones_to_add))
        #     end
        # else
        #     println("setting err in birth")
        #     ER_AFTER_DEL = error_gram(tr)
        #     ANALYSIS = (noises_to_add, tones_to_add)
        # end

        possibilities = [
            ((:noise, params) for params in noises_to_add)...,
            ((:tone, params) for params in tones_to_add)... #,
            # ((:delete, :noise, params) for params in noises_to_delete)...,
            # ((:delete, :tone, params) for params in tones_to_delete)...
        ]

        smartprior = length(possibilities) == 0 ? 0. : 0.95

        do_smart ~ bernoulli(smartprior)
        if do_smart
            scores = score_possibilities(possibilities)

            type_regions = collect(unique((type, region) for (type, (region, _)) in possibilities))
            
            type_region_to_indices = Dict()
            for i=1:length(possibilities)
                (typ, (reg, _)) = possibilities[i]
                if haskey(type_region_to_indices, (typ, reg))
                    push!(type_region_to_indices[(typ, reg)], i)
                else
                    type_region_to_indices[(typ, reg)] = [i]
                end
            end

            type_region_scores = [
                sum(
                    scores[i] for i in type_region_to_indices[(typ, region)]
                )
                for (typ, region) in type_regions
            ]

            if !isapprox(sum(type_region_scores), 1)
                println("scores: ", type_region_scores)
            end
            type_region ~ list_categorical(type_regions, type_region_scores)
            (typ, region) = type_region

            possible_indices = type_region_to_indices[type_region]
            if typ === :tone
                tone_scores = [scores[i] for i in possible_indices]
                tone_idx ~ categorical(tone_scores/sum(tone_scores))
                (_, (_, rb)) = possibilities[possible_indices[tone_idx]]
                erb ~ normal(rb, ERB_STD)
            else
                (_, (_, mp)) = possibilities[possible_indices[1]]
                amp ~ normal(mp, AMP_STD)
            end

            regionsize = size(error_gram(tr))[2]
            scaled_region = (region[1]/regionsize, region[2]/regionsize)
            onset ~ shifted_folded_normal(0., scaled_region[1], STARTTIME_STD)
            duration ~ shifted_folded_normal(0.1, scaled_region[2] - scaled_region[1], DURATION_STD)
        else
            # regenerate everything from the prior!
        end
    else
        death_idx ~ uniform_discrete(1, tr[:kernel => :n_tones])

        # abst = GenWorldModels.convert_to_abstract(tr.world, AudioSource(death_idx))

        is_noise = tr[:world => :waves => AudioSource(death_idx) => :is_noise]
        
        (scene_length, steps, sr, wts, gtg_params) = get_args(tr)
        n_samples = Int(floor(scene_length * sr))
        waves = (tr[:world => :waves => AudioSource(i)] for i=1:tr[:kernel => :n_tones] if i != death_idx)
        underlying_waves_without_deletion = reduce(+, waves; init=zeros(n_samples))
        underlying_gram, = gammatonegram(underlying_waves_without_deletion, wts, sr, gtg_params)
        err_after_deletion = observed_gram(tr) - underlying_gram

        (detected_noises, detected_tones) = analyze_gram(err_after_deletion)

        # if ER_AFTER_DEL === nothing
        #     println("setting err on delete")
        #     global ER_AFTER_DEL = err_after_deletion
        #     global ANALYSIS = (detected_noises, detected_tones)
        # else
        #     if !isapprox(ER_AFTER_DEL, err_after_deletion)
        #         display((ER_AFTER_DEL - err_after_deletion)[30:40, 90:110])
        #     end
        #     if !((detected_noises, detected_tones) == ANALYSIS)
        #         println("birth got ", ANALYSIS, "; death got ", (detected_noises, detected_tones))
        #     end
        # end


        if (is_noise && length(detected_noises) == 0) || (!is_noise && length(detected_tones) == 0)
            rev_smart_prior = 0
        else
            rev_smart_prior = 0.95
        end
        rev_smart ~ bernoulli(rev_smart_prior)
        if rev_smart && rev_smart_prior == 0
            println("prior is 0 but chose smart!")
        end

        if rev_smart
            regions = [region for (region, _) in (is_noise ? detected_noises : detected_tones)]
            # println("deathrev $(is_noise ? "noise" : "tone") regions : ", regions)
            
            start = tr[:world => :waves => AudioSource(death_idx) => :onset]
            dur = tr[:world => :waves => AudioSource(death_idx) => :duration]
            this_region = (start, start + dur)
            overlaps = [overlap(region, this_region) for region in regions]
            probs = overlaps .+ 1
            probs = probs/sum(probs)
            region ~ uniform_from_list(regions, probs)

            if !is_noise
                # println("region: $region")
                # println("detected tones: $detected_tones")
                num_tones = length([r for (r, _) in detected_tones if r == region])
                tone_idx ~ uniform_discrete(1, num_tones)
            end
        end
    end
end

@oupm_involution smart_bd_inv (old, fwd) to (new, bwd) begin
    do_birth = @read(fwd[:do_birth], :disc)
    current_n_tones = @read(old[:kernel => :n_tones], :disc)
    @write(bwd[:do_birth], !do_birth, :disc)
    if do_birth
        idx = @read(fwd[:birth_idx], :disc)
        src = AudioSource(idx)
        @birth(src)
        @write(new[:kernel => :n_tones], current_n_tones + 1, :disc)

        is_smart = @read(fwd[:do_smart], :disc)
        if is_smart
            (type, region) = @read(fwd[:type_region], :disc)
            is_noise = type === :noise
            @write(new[:world => :waves => src => :is_noise], is_noise, :disc)
            @copy(fwd[:onset], new[:world => :waves => src => :onset])
            @copy(fwd[:duration], new[:world => :waves => src => :duration])
            if is_noise
                @copy(fwd[:amp], new[:world => :waves => src => :amp])
            else
                @copy(fwd[:erb], new[:world => :waves => src => :erb])
            end
        else
            # properties will be regenerated from the prior
            @regenerate(:world => :waves => src)
        end

        # reverse move:
        @write(bwd[:death_idx], idx, :disc)
        @write(bwd[:rev_smart], is_smart, :disc)
        if is_smart
            @write(bwd[:region], region, :disc)
            if !is_noise
                @write(bwd[:tone_idx], @read(fwd[:tone_idx], :disc), :disc)
            end
        end
    else
        idx = @read(fwd[:death_idx], :disc)
        src = AudioSource(idx)
        @death(src)
        @write(new[:kernel => :n_tones], current_n_tones - 1, :disc)

        # set up reverse move
        rev_smart = @read(fwd[:rev_smart], :disc)
        if rev_smart
            # in this case, we should be guaranteed that there is a valid reversing smart move
            region = @read(fwd[:region], :disc)
            is_noise = @read(old[:world => :waves => src => :is_noise], :disc)
            if is_noise
                @write(bwd[:type_region], (:noise, region), :disc)
            else
                @write(bwd[:type_region], (:tone, region), :disc)
                @copy(fwd[:tone_idx], bwd[:tone_idx])
            end

            # write properties bwd!
            @copy(old[:world => :waves => src => :onset], bwd[:onset])
            @copy(old[:world => :waves => src => :duration], bwd[:duration])
            if is_noise
                @copy(old[:world => :waves => src => :amp], bwd[:amp])
            else
                @copy(old[:world => :waves => src => :erb], bwd[:erb])
            end
        else
            # if the birth move is dumb, we regenerate everything in that direction
            @save_for_reverse_regenerate(:world => :waves => src)
        end
        @write(bwd[:do_smart], rev_smart, :disc)
        @write(bwd[:birth_idx], idx, :disc)
    end
end

smart_bd_mh_kern = OUPMMHKernel(smart_bd_proposal, (), smart_bd_inv)

function smart_bd_inference_iter(tr)
    tr = generic_no_num_change_inference_iter(tr)
    tr, _ = mh(tr, smart_bd_mh_kern; check=false)
    return tr
end
function do_smart_bd_inference(tr, iters, record_iter!)
    for i = 1:iters
        tr = smart_bd_inference_iter(tr)
        record_iter!(tr)
    end
    return tr
end

include("smart_sm.jl")

include("drift.jl")
function smart_bd_drift_inference_iter(tr)
    tr = drift_inference_iter(tr)
    tr, _ = mh(tr, smart_bd_mh_kern; check=false)
    return tr
end
function do_smart_bd_drift_inference(tr, iters, record_iter!)
    for i = 1:iters
        tr = smart_bd_drift_inference_iter(tr)
        record_iter!(tr)
    end
    return tr
end

function smart_smbd_drift_inference_iter(tr)
    tr = drift_inference_iter(tr)
    tr, _ = mh(tr, smart_bd_mh_kern; check=false)
    tr, _ = mh(tr, smart_splitmerge_mh_kern; check=false)
    return tr
end
function do_smart_drift_smbd_inference(tr, iters, record_iter!)
    for i = 1:iters
        tr = smart_smbd_drift_inference_iter(tr)
        record_iter!(tr)
    end
    return tr
end