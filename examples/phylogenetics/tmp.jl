function f(x, y)
    return x + y
end

@gen function noisy_observe_sum()
    value1 ~ normal(0, 1)
    value2 ~ normal(0, 1)
    summed = f(x, y)
    observation ~ normal(summed, 0.3)
    return observation
end

# This proposal says that we will update value1 and value2 so that:
# value1 = split × new_total
# value2 = (1 - split) × new_total
@gen function make_decisions_for_proposal(tr)
    split ~ beta(1, 1)
    new_total ~ normal(tr[:observation], 0.3)
end



@transform apply_proposal_and_find_reverse_move (old_model, fwd_proposal) to (new_model, bwd_proposal) begin
    new_total, split = @read(fwd_proposal[:new_total]), @read(fwd_proposal[:split])
    @write(new_model[:value1], new_total * split)
    @write(new_model[:value2], new_total * (1 - split))

    old_v1, old_v2 = @read(old_model[:value1]), @read(old_model[:value2])
    @write(bwd_proposal[:split], old_v1/(old_v1 + old_v2))
    @write(bwd_proposal[:new_total], old_v1 + old_v2)
end

new_tr, was_accepted = mh(old_tr,
        make_decisions_for_proposal,
        apply_proposal_and_find_reverse_move
    )
