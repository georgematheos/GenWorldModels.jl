function get_likely_start_end(tr, ch)
    eg = error_gram(tr)
    (ysize, xsize) = size(eg)
    if ch[:is_noise]
        idx = Int(floor(ysize/2))
    else
        idx = Int(floor(pos_for_erb_val(ch[:erb])))
    end
    # println("onset = ", ch[:onset])
    # println("thresholded (onset + duration)*xsize = ", (min(ch[:onset] + ch[:duration], 1)) * xsize)
    # println("xsize: ", xsize)
    st = Int(floor(ch[:onset] * xsize))
    nd = Int(floor((min(ch[:onset] + ch[:duration], 1)) * xsize))
    # println("st, end = ", (st, nd))
    row = error_gram(tr)[idx, st:nd]
    if isempty(row)
        return nothing
    end
    regions = get_regions(get_thresholded(row, 0.2))
    # println("regions: ", regions)
    if isempty(regions)
        return nothing
    end
    areas = map(((a, b),) -> b - a, regions)
    maxidx = findmax(areas)[2]
    maxregion = regions[maxidx]
    if maxregion[2]-maxregion[1] < 2
        return nothing
    end
    return (st + maxregion[1], st + maxregion[2])
end

@gen function smart_splitmerge_prop(tr)
    n_tones = tr[:kernel => :n_tones]
    merge_possible = (
      n_tones > 1 &&
      (length([idx for idx = 1:n_tones if tr[:world => :waves => AudioSource(idx) => :is_noise]]) > 1 ||
        length([idx for idx = 1:n_tones if !tr[:world => :waves => AudioSource(idx) => :is_noise]]) > 1)
    )
    param = merge_possible ? 0.5 : 1.
    do_split ~ bernoulli(param)
                      
    if do_split
        solo_idx ~ uniform_discrete(1, n_tones)
        deuce_idx1 ~ uniform_discrete(1, n_tones + 1)
        deuce_idx2 ~ uniform_discrete(1, n_tones + 1)
        if deuce_idx1 != deuce_idx2
            ch = get_submap(get_choices(tr), :world => :waves => AudioSource(solo_idx))
            if !ch[:is_noise]
                erb1 ~ normal(ch[:erb], .5)
                erb2 ~ normal(ch[:erb], .5)
            else
                amp1 ~ normal(ch[:amp], .5)
                amp2 ~ normal(ch[:amp], .5)
            end
            # the split sounds go from ch[:onset] to ch[:onset] + dur1, and
            # startpoint to ch[:onset]+ch[:duration]-dur2 to ch[:onset]+ch[:duration]

            likelies = get_likely_start_end(tr, ch)
            if likelies !== nothing
                regionsize = size(error_gram(tr))[2]
                likely_start, likely_end = (likelies[1]/regionsize, likelies[2]/regionsize)
                # println("ls, le are ", likely_start, ", ", likely_end)
                likely_dur_1 = likely_start - ch[:onset]
                likely_dur_2 = (ch[:onset] + ch[:duration]) - likely_end
                # println("onset, duration are ", ch[:onset], "; ", ch[:duration])

                dur1 ~ shifted_folded_normal(0.1, likely_dur_1, 0.05)
                dur2 ~ shifted_folded_normal(0.1, likely_dur_2, 0.05)
            else
                dur1 ~ uniform(0.1, max(.11, 0.7 * ch[:duration]))
                dur2 ~ uniform(0.1, max(.11, 0.7 * ch[:duration]))    
            end
        end
    else
        solo_idx ~ uniform_discrete(1, max(1, n_tones - 1))
        deuce_idx1 ~ uniform_discrete(1, n_tones)
        if (deuce_idx1 > n_tones)
          # if this happens, this is the backward step for an impossible forward move;
          # just escape the function quickly in this case
            deuce_idx2 ~ uniform_discrete(1, n_tones)
            return nothing
        end
        ch1 = get_submap(get_choices(tr), :world => :waves => AudioSource(deuce_idx1))
        compatible_indices = [
            idx for idx = 1:n_tones
            if (
                tr[:world => :waves => AudioSource(idx) => :is_noise] == ch1[:is_noise] && 
                tr[:world => :waves => AudioSource(idx) => :onset] >= ch1[:onset]
            )
        ]
        deuce_idx2 ~ uniform_from_list(compatible_indices)
        ch2 = get_submap(get_choices(tr), :world => :waves => AudioSource(deuce_idx2))
        
        if deuce_idx1 != deuce_idx2
            if ch1[:is_noise]
                amp ~ normal((ch1[:amp] + ch2[:amp]) / 2, .5)
            else
                erb ~ normal((ch1[:erb] + ch2[:erb]) / 2, .5)
            end    
        end
    end
end

smart_splitmerge_mh_kern = OUPMMHKernel(smart_splitmerge_prop, (), splitmerge_inv)

function smart_smbd_inference_iter(tr)
    tr = generic_no_num_change_inference_iter(tr)
    tr, _ = mh(tr, smart_bd_mh_kern; check=false)
    tr, _ = mh(tr, smart_splitmerge_mh_kern; check=false)
    return tr
end
function do_smart_smbd_inference(tr, iters, record_iter!)
    for i = 1:iters
        tr = smart_smbd_inference_iter(tr)
        record_iter!(tr)
    end
    return tr
end
