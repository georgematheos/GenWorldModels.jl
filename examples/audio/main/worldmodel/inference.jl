function generic_no_num_change_inference_iter(tr)
    for j = 1:tr[:kernel => :n_tones]
        tr, _ = mh(tr, select(:world => :waves => AudioSource(j)))
        if tr[:world => :waves => AudioSource(j) => :is_noise]
            tr, _ = mh(tr, select(:world => :waves => AudioSource(j) => :amp))
        else
            tr, _ = mh(tr, select(:world => :waves => AudioSource(j) => :erb))
        end
        tr, _ = mh(tr, select(:world => :waves => AudioSource(j) => :onset))
        tr, _ = mh(tr, select(:world => :waves => AudioSource(j) => :duration))
    end
    return tr
end
  
function generic_inference_iter(tr)
    tr = generic_no_num_change_inference_iter(tr)
    tr, _ = mh(tr, select(:kernel => :n_tones))
    return tr
end
  
function do_generic_inference(tr, iters, record_iter!)
    for i = 1:iters
        tr = generic_inference_iter(tr)
        record_iter!(tr)
    end
    return tr
end

### BIRTH/DEATH ###
@gen function birth_death_proposal(tr)
    do_birth ~ bernoulli(0.5)
    if do_birth
        idx ~ uniform_discrete(1, tr[:kernel => :n_tones] + 1)
    else
        idx ~ uniform_discrete(1, tr[:kernel => :n_tones])
    end
end
  
  
@oupm_involution birth_death_inv (old_tr, fwd_prop_tr) to (new_tr, bwd_prop_tr) begin
    do_birth = @read(fwd_prop_tr[:do_birth], :disc)
    idx = @read(fwd_prop_tr[:idx], :disc)
    num = @read(old_tr[:kernel => :n_tones], :disc)
    if do_birth
        @birth(AudioSource(idx))
        @write(new_tr[:kernel => :n_tones], num + 1, :disc)
        @regenerate(:world => :waves => AudioSource(idx))
    else
        @death(AudioSource(idx))
        @write(new_tr[:kernel => :n_tones], num - 1, :disc)
        @save_for_reverse_regenerate(:world => :waves => AudioSource(idx))
    end
    @write(bwd_prop_tr[:do_birth], !do_birth, :disc)
    @write(bwd_prop_tr[:idx], idx, :disc)
end
  
function birth_death_inference_iter(tr)
    tr = generic_no_num_change_inference_iter(tr)
    tr, _ = mh(tr, birth_death_mh_kern)
    return tr
end
function do_birth_death_inference(tr, iters, record_iter!)
    for i = 1:iters
        tr = birth_death_inference_iter(tr)
        record_iter!(tr)
    end
    return tr
end

birth_death_mh_kern = OUPMMHKernel(birth_death_proposal, (), birth_death_inv)

### SPLIT/MERGE ###

@dist function uniform_from_list(list)
    idx = uniform_discrete(1, length(list))
    list[idx]
end

@gen function splitmerge_prop(tr)
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
            dur1 ~ uniform(0.1, max(.11, 0.7 * ch[:duration]))
            dur2 ~ uniform(0.1, max(.11, 0.7 * ch[:duration]))
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
splitmerge_mh_kern = OUPMMHKernel(splitmerge_prop, (), splitmerge_inv)

function split_merge_inference_iter(tr; num_sm_per_iter=4)
    tr, _ = mh(tr, birth_death_mh_kern)
    tr = generic_no_num_change_inference_iter(tr)
    for _ = 1:num_sm_per_iter
        new_tr, sm_acc = mh(tr, splitmerge_mh_kern; check=false)
        # if sm_acc && tr[:kernel => :n_tones] != new_tr[:kernel => :n_tones]
        #     is_split = tr[:kernel => :n_tones] < new_tr[:kernel => :n_tones]
        #     println("SUCCESSFUL SM MOVE: $(is_split ? "split" : "merge")")
        # end
        tr = new_tr
    end
    return tr
end
function do_split_merge_inference(tr, iters, record_iter!; num_sm_per_iter=4)
    for i = 1:iters
        tr = split_merge_inference_iter(tr; num_sm_per_iter)
        record_iter!(tr)
    end
    return tr
end