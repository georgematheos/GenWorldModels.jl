function get_random_address(tr)
    get_random_address(get_submap(get_choices(tr), :world))
end
function get_random_address(choices::ChoiceMap)
    subs = collect(get_submaps_shallow(choices))
    idx = uniform_discrete(1, length(subs))
    addrstart, submap = subs[idx]
    if has_value(submap)
        return addrstart
    else
        return addrstart => get_random_address(submap)
    end
end

function regen_mh_iter(tr)
    a = get_random_address(tr)
    tr, acc = mh(tr, select(a))
    return tr
end

function run_regen_inference(tr, num_iters, examine; examine_freq)
    for i=1:num_iters
        tr = regen_mh_iter(tr)
        if i % examine_freq == 0
            examine(tr)
        end
    end
end