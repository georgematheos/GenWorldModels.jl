function get_likely_start_end(tr, ch)
    eg = error_gram(tr)
    (ysize, xsize) = size(eg)

    st = max(1, Int(floor(ch[:onset] * xsize)))
    nd = Int(floor((min(ch[:onset] + ch[:duration], 1)) * xsize))

    if nd - st < 1
        return nothing
    end

    if ch[:is_noise]
        miny, maxy = 1, ysize
    else
        meany = Int(floor(Detector.pos_for_erb_val(ch[:erb])))
        miny, maxy = max(1, Int(meany - TONESIZE/2)), min(Int(meany + TONESIZE/2), ysize)
    end

    eg_region = eg[miny:maxy, st:nd]
    
    (startsegs, endsegs) = Detector.get_start_end_segs(eg_region)
    if isempty(startsegs) || isempty(endsegs)
        return nothing
    end

    selected_startseg = startsegs[argmax(map(Detector.seglength, startsegs))]
    selected_endseg = endsegs[argmax(map(Detector.seglength, endsegs))]

    (st + selected_endseg.x, st + selected_startseg.x)
end

@gen function smart_splitmerge_prop(tr)
    n_tones = tr[:kernel => :n_tones]
    merge_possible = (
      n_tones > 1 &&
      (length([idx for idx = 1:n_tones if tr[:world => :waves => AudioSource(idx) => :is_noise]]) > 1 ||
        length([idx for idx = 1:n_tones if !tr[:world => :waves => AudioSource(idx) => :is_noise]]) > 1)
    )
    splitprob = merge_possible ? (n_tones == 4 ? 0. : 0.5) : 1.
    do_split ~ bernoulli(splitprob)
                      
    if do_split
        solo_idx ~ uniform_discrete(1, n_tones)
        deuce_idx1 ~ uniform_discrete(1, n_tones + 1)
        deuce_idx2 ~ uniform_discrete(1, n_tones + 1)
        if deuce_idx1 != deuce_idx2
            ch = get_submap(get_choices(tr), :world => :waves => AudioSource(solo_idx))
            if !ch[:is_noise]
                erb1 ~ truncated_normal(ch[:erb], ERB_STD, MIN_ERB, MAX_ERB)
                erb2 ~ truncated_normal(ch[:erb], ERB_STD, MIN_ERB, MAX_ERB)
            else
                amp1 ~ normal(ch[:amp], AMP_STD)
                amp2 ~ normal(ch[:amp], AMP_STD)
            end
            # the split sounds go from ch[:onset] to ch[:onset] + dur1, and
            # startpoint to ch[:onset]+ch[:duration]-dur2 to ch[:onset]+ch[:duration]

            mindur = MIN_DURATION(scene_length)
            likelies = get_likely_start_end(tr, ch)
            if likelies !== nothing
                regionsize = size(error_gram(tr))[2]
                likely_start, likely_end = (likelies[1]/regionsize, likelies[2]/regionsize)
                # println("ls, le are ", likely_start, ", ", likely_end)
                likely_dur_1 = likely_start - ch[:onset]
                likely_dur_2 = (ch[:onset] + ch[:duration]) - likely_end
                # println("onset, duration are ", ch[:onset], "; ", ch[:duration])

                dur1 ~ truncated_normal(likely_dur_1, DURATION_STD, mindur, MAX_DURATION(scene_length))
                dur2 ~ truncated_normal(likely_dur_2, DURATION_STD, mindur, MAX_DURATION(scene_length))
            else
                dur1 ~ uniform(mindur, max(mindur + .01, 0.7 * ch[:duration]))
                dur2 ~ uniform(mindur, max(mindur + .01, 0.7 * ch[:duration]))    
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
                amp ~ normal((ch1[:amp] + ch2[:amp]) / 2, AMP_STD)
            else
                erb ~ normal((ch1[:erb] + ch2[:erb]) / 2, ERB_STD)
            end    
        end
    end
end

@oupm_involution splitmerge_inv (old_tr, fwd_prop_tr) to (new_tr, bwd_prop_tr) begin
    n_tones = @read(old_tr[:kernel => :n_tones], :disc)
    do_split = @read(fwd_prop_tr[:do_split], :disc)
    deuce_idx1 = @read(fwd_prop_tr[:deuce_idx1], :disc)
    deuce_idx2 = @read(fwd_prop_tr[:deuce_idx2], :disc)
    solo_idx = @read(fwd_prop_tr[:solo_idx], :disc)
    if deuce_idx1 != deuce_idx2
        if do_split
            @split(AudioSource(solo_idx), deuce_idx1, deuce_idx2)
            @write(new_tr[:kernel => :n_tones], n_tones + 1, :disc)

            o(x) = :world => :waves => AudioSource(solo_idx) => x
            n1(x) = :world => :waves => AudioSource(deuce_idx1) => x
            n2(x) = :world => :waves => AudioSource(deuce_idx2) => x

            # copy is noise
            @copy(old_tr[o(:is_noise)], new_tr[n1(:is_noise)])
            @copy(old_tr[o(:is_noise)], new_tr[n2(:is_noise)])

            # start and end times
            @copy(old_tr[o(:onset)], new_tr[n1(:onset)])
            @copy(fwd_prop_tr[:dur1], new_tr[n1(:duration)])
            @copy(fwd_prop_tr[:dur2], new_tr[n2(:duration)])

            old_ons = @read(old_tr[o(:onset)], :cont)
            old_dur = @read(old_tr[o(:duration)], :cont)
            dur2 = @read(fwd_prop_tr[:dur2], :cont)
            @write(new_tr[n2(:onset)], old_ons + old_dur - dur2, :cont)

            # amp/erb
            if @read(old_tr[o(:is_noise)], :disc)
                @copy(fwd_prop_tr[:amp1], new_tr[n1(:amp)])
                @copy(fwd_prop_tr[:amp2], new_tr[n2(:amp)])
                @copy(old_tr[o(:amp)], bwd_prop_tr[:amp])
            else
                @copy(fwd_prop_tr[:erb1], new_tr[n1(:erb)])
                @copy(fwd_prop_tr[:erb2], new_tr[n2(:erb)])
                @copy(old_tr[o(:erb)], bwd_prop_tr[:erb])
            end
        else
            @merge(AudioSource(solo_idx), deuce_idx1, deuce_idx2)
            @write(new_tr[:kernel => :n_tones], n_tones - 1, :disc)
            
            n(x) = :world => :waves => AudioSource(solo_idx) => x
            o1(x) = :world => :waves => AudioSource(deuce_idx1) => x
            o2(x) = :world => :waves => AudioSource(deuce_idx2) => x

            # is_noise
            @copy(old_tr[o1(:is_noise)], new_tr[n(:is_noise)])

            # onset & duration
            start1 = @read(old_tr[o1(:onset)], :cont)
            dur1 = @read(old_tr[o1(:duration)], :cont)
            start2 = @read(old_tr[o2(:onset)], :cont)
            dur2 = @read(old_tr[o2(:duration)], :cont)
            end2 = start2 + dur2
            full_dur = end2 - start1

            @copy(old_tr[o1(:onset)], new_tr[n(:onset)])
            @write(new_tr[n(:duration)], full_dur, :cont)

            @write(bwd_prop_tr[:dur1], dur1, :cont)
            @write(bwd_prop_tr[:dur2], dur2, :cont)

            if @read(old_tr[o1(:is_noise)], :disc)
                @copy(old_tr[o1(:amp)], bwd_prop_tr[:amp1])
                @copy(old_tr[o2(:amp)], bwd_prop_tr[:amp2])
                @copy(fwd_prop_tr[:amp], new_tr[n(:amp)])
            else
                @copy(old_tr[o1(:erb)], bwd_prop_tr[:erb1])
                @copy(old_tr[o2(:erb)], bwd_prop_tr[:erb2])
                @copy(fwd_prop_tr[:erb], new_tr[n(:erb)])
            end
        end
    end
    @write(bwd_prop_tr[:do_split], !do_split, :disc)
    @copy(fwd_prop_tr[:solo_idx], bwd_prop_tr[:solo_idx])
    @copy(fwd_prop_tr[:deuce_idx1], bwd_prop_tr[:deuce_idx1])
    @copy(fwd_prop_tr[:deuce_idx2], bwd_prop_tr[:deuce_idx2])
end

smart_splitmerge_mh_kern = OUPMMHKernel(smart_splitmerge_prop, (), splitmerge_inv)