function run_inference(tr, iters)
    for i=1:iters
        tr = run_inference_iter(tr)
    end
end
function run_inference_iter(tr)
    tr, _ = mh(tr, splitmerge_prop, splitmerge_inv)
    tr = regenerate_vals(tr)
    tr, _ = mh(tr, birthdeath_prop, birthdeath_inv)
    tr = regenerate_vals(tr)
    tr, _ = mh(tr, swap_prop, swap_inv)
    tr = regenerate_vals(tr)
end
function regenerate_vals(tr)
    for _=1:5
        v = uniform_from_set(all_instantiated_of_type((tr.world, :Virus)))
        tr, _ = mh(tr, select(:world => :genome => v))
    end
end

@gen function splitmerge_prop(tr)
    viruses = collect(all_instantiated_of_type(tr.world, Virus))
    nonobserved_viruses = filter(x -> tr[:world => :is_observed => x], viruses)
    num_children = map(x -> tr[:world => :num_children => (x,)], nonobserved_viruses)
    virus ~ set_categorical(nonobserved_viruses, renormalized)

    do_split ~ bernoulli(0.5)
    if do_split
        renormalized = num_children / sum(num_children)

        parent = virus.origin[1]
        num_siblings = tr[:world => :num_children => (parent,)] - 1
        
        new_idx1 ~ uniform_discrete(1, num_siblings + 2)
        new_idx2 ~ uniform_discrete(1, num_siblings + 2)

        num_children = tr[:world => :num_children => (virus,)]
        for i=1:num_children
            {:to_first => i} ~ bernoulli(0.5)
        end
    else
        num_children = tr[:world => :num_children => (virus,)]

        # merge the following 2 children:
        child1_idx ~ uniform_discrete(1, num_children)
        child2_idx ~ uniform_discrete(1, num_children)


    end

end









@oupm_involution splitmerge_inv (old, fwd) to (new, bwd) begin
    do_split = @read(fwd[:do_split], :disc)
    if do_split
        from = @read(fwd[:virus], :disc)
        to_idx1 = @read(fwd[:new_idx1], :disc)
        to_idx2 = @read(fwd[:new_idx2], :disc)
       
        num_children = !read(old[:world => :num_children => (from,)], :disc)
        moves = [
            Virus(from.origin, i) => Virus(from.origin, (fwd[:to_first => i] ? to_idx1 : to_idx2))
            for i=1:num_children
        ]

        @split(from, to_idx1, to_idx2, moves)
        @regenerate(:world => :genome => Virus(from.origin, to_idx1))
        @regenerate(:world => :genome => Virus(from.origin, to_idx2))

        @save_for_reverse_regenerate(:world => :genome => from)
        @write(bwd[:do_split], false, :disc)
        @write(bwd[:])
    end
end
