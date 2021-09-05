include("split_merge.jl")
include("gibbs_updates.jl")

split_merge(tr) = mh(tr, MHProposal(split_merge_kernel); check=false)[1]# check=true, roundtrip_atol=0.5)[1]
function w_move(tr)
    a = false
    for _=1:4
        tr, acc = mh(tr, MHProposal(update_w); check=false)
        a = a || acc
    end
    if !a
        @warn "w move not accepted"
    end
    # @assert acc "w was not gibbs"
    return tr
end
gibbs(proposal, update_name) = tr ->
    let (newtr, acc) = mh(tr, proposal, ())
        if !acc
            @warn "move for $update_name was rejected (should be gibbs)"
            error()
        end
        newtr
    end

function inference_cycle(tr)
    tr = w_move(tr)
    tr = gibbs(update_means, "means")(tr)
    tr = gibbs(update_vars, "vars")(tr)
    tr = gibbs(update_allocations, "allocations")(tr)
    tr = split_merge(tr)
    return tr
end
function do_inference(tr, n_iters; get_map=false)
    map = tr
    for i=1:n_iters
        tr = inference_cycle(tr)
        map = get_score(map) > get_score(tr) ? map : tr
    end
    return get_map ? map : tr
end