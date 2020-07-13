### diff propagation ###

function Base.:(:)(first::Diffed{Int}, last::Diffed{Int})
    is_no_change = get_diff(first) === NoChange() && get_diff(last) === NoChange()
    Diffed(strip_diff(first):strip_diff(last), is_no_change ? NoChange() : UnknownChange())
end
Base.:(:)(first::Int, last::Diffed{Int}) = Diffed(first, NoChange()):last
Base.:(:)(first::Diffed{Int}, last::Int) = first:Diffed(last, NoChange())

function worldmap_args(world, addr::Symbol, itr)
    [world[addr][x] for x in itr]
end
function worldmap_args(world::Diffed, addr::Symbol, itr::Diffed{<:Any, NoChange})
    diffed_mgfcalls = (world[addr][x] for x in strip_diff(itr))
    mgfcalls = [strip_diff(call) for call in diffed_mgfcalls]
    mgfcall_diffs = [get_diff(call) for call in diffed_mgfcalls]
    diffed_inds = findall(diff -> diff !== NoChange(), mgfcall_diffs)

    diff = VectorDiff(length(mgfcalls), length(mgfcalls), Dict(ind => mgfcall_diffs[ind] for ind in diffed_inds))
    Diffed(mgfcalls, diff)
end
worldmap_args(world::Diffed, addr::Symbol, itr) = worldmap_args(world, addr, Diffed(itr, NoChange()))
worldmap_args(world::Diffed, addr::Symbol, itr::Diffed{<:Any, UnknownChange}) = worldmap_args(strip_diff(world), addr, itr)
function worldmap_args(world, addr::Symbol, itr::Diffed{<:Any, UnknownChange})
    calls = [world[addr][x] for x in strip_diff(itr)]
    Diffed(calls, UnknownChange())
end

### model ###

@gen (static, diffs) function generate_coin(world, coin_idx)
    beta_prior ~ lookup_or_generate(world[:args][:beta_prior])
    (a, b) = beta_prior
    prob ~ beta(a, b)
    return prob
end

@gen (static, diffs) function flips_for_coin(world, coin_idx)
    nfpc ~ lookup_or_generate(world[:args][:num_flips_per_coin])
    weight ~ lookup_or_generate(world[:coin][coin_idx])
    flips ~ binom(nfpc, weight)
    return flips
end

@gen (static, diffs) function _flip_coins(world, num_coins)
    flips ~ Map(lookup_or_generate)(worldmap_args(world, :flips, 1:num_coins))
    return flips
end

# so args = (num_flips_per_coin, beta_prior, num_coins)
flip_coins = UsingWorld(
    _flip_coins,
    :coin => generate_coin,
    :flips => flips_for_coin;
    world_args=(:num_flips_per_coin, :beta_prior)
)
@load_generated_functions()

### testing ###

@testset "world args" begin
    almost_always_heads_prior = (5000, 1) # should result in coins with extremely high probability of heads
    tr = simulate(flip_coins, (2, almost_always_heads_prior, 5))
    @test get_retval(tr) == fill(2, 5) # each coin should have 5 heads (unless this is an extremely rare event!)
    @test get_args(tr) == (2, almost_always_heads_prior, 5)

    tr, weight = generate(flip_coins, (2, (1, 1), 5), choicemap(
        (:world => :coin => 1 => :prob, 0.7),
        (:world => :coin => 2 => :prob, 0.7),
        (:world => :coin => 3 => :prob, 0.7),
        (:world => :coin => 4 => :prob, 0.7),
        (:world => :coin => 5 => :prob, 0.7)
    ))
    @test isapprox(weight, 5*logpdf(beta, 0.7, 1, 1))

    # `update` by changing a world arg but nothing else
    new_tr, weight, retdiff, discard = update(tr, (2, (1, 2), 5), (NoChange(), UnknownChange(), NoChange()), EmptyChoiceMap())
    expected_score_delta = 5*(logpdf(beta, 0.7, 1, 2) - logpdf(beta, 0.7, 1, 1))
    @test isapprox(expected_score_delta, get_score(new_tr) - get_score(tr))
    @test isapprox(weight, expected_score_delta)
    @test retdiff == NoChange()
    @test all(new_tr[:world => :coin => i] == tr[:world => :coin => i] for i=1:5)
    @test discard == EmptyChoiceMap()

    # `regenerate` by changing world args but nothing else
    new_tr, weight, retdiff = regenerate(tr, (2, (1, 2), 5), (NoChange(), UnknownChange(), NoChange()), EmptySelection())
    @test isapprox(expected_score_delta, get_score(new_tr) - get_score(tr))
    @test isapprox(weight, expected_score_delta)
    @test retdiff == NoChange()
    @test all(new_tr[:world => :coin => i] == tr[:world => :coin => i] for i=1:5)

    # `update` world args and world choices
    new_tr, weight, retdiff, discard = update(
        tr,
        (2, (1, 2), 5),
        (NoChange(), UnknownChange(), NoChange()),
        choicemap((:world => :coin => 5 => :prob, 0.25))
    )
    num_heads_5 = tr[:kernel => :flips => 5]
    expected_score_delta = (
        logpdf(beta, 0.25, 1, 2) + 4*logpdf(beta, 0.7, 1, 2) - 5*logpdf(beta, 0.7, 1, 1) # coin generation score change
        + logpdf(binom, num_heads_5, 2, 0.25) - logpdf(binom, num_heads_5, 2, 0.7)
    )
    @test isapprox(expected_score_delta, get_score(new_tr) - get_score(tr))
    @test isapprox(weight, expected_score_delta)
    @test retdiff == NoChange()
    @test all(new_tr[:world => :coin => i] == tr[:world => :coin => i] for i=1:4)
    @test get_retval(new_tr) == get_retval(tr)
    @test retdiff == NoChange()
    @test discard == choicemap((:world => :coin => 5 => :prob, 0.7))

    # `regenerate` world args and world choices
    new_tr, weight, retdiff = regenerate(
        tr,
        (2, (1, 2), 5),
        (NoChange(), UnknownChange(), NoChange()),
        select(:world => :coin => 5)
    )
    num_heads_5 = tr[:world => :flips => 5]
    new_prob_5 = new_tr[:world => :coin => 5]
    selected_coingen_score_change = logpdf(beta, new_prob_5, 1, 2) - logpdf(beta, 0.7, 1, 1)
    nonselected_coingen_score_change = 4*(logpdf(beta, 0.7, 1, 2) - logpdf(beta, 0.7, 1, 1))
    flips_score_change = logpdf(binom, num_heads_5, 2, new_prob_5) - logpdf(binom, num_heads_5, 2, 0.7)
    expected_score_delta = selected_coingen_score_change + nonselected_coingen_score_change + flips_score_change
    @test isapprox(expected_score_delta, get_score(new_tr) - get_score(tr))
    @test isapprox(weight, nonselected_coingen_score_change + flips_score_change)
    @test retdiff == NoChange()
    @test all(new_tr[:world => :coin => i] == tr[:world => :coin => i] for i=1:4)
    @test new_tr[:world => :coin => 5] != tr[:world => :coin => 5]
    @test get_retval(new_tr) == get_retval(tr)
    @test retdiff == NoChange()

    # `update` multiple world args, a kernel arg, and a value
    new_tr, weight, retdiff, discard = update(
        tr,
        (3, (1, 2), 4),
        (UnknownChange(), UnknownChange(), UnknownChange()),
        choicemap((:world => :coin => 1 => :prob, 0.25))
    )
    num_heads_1 = tr[:world => :flips => 1]
    expected_score_delta = (
        logpdf(beta, 0.25, 1, 2) + 3*logpdf(beta, 0.7, 1, 2) - 5*logpdf(beta, 0.7, 1, 1) # coin generation score change
        + logpdf(binom, num_heads_1, 3, 0.25) - logpdf(binom, num_heads_1, 2, 0.7) # prob of num_flips for coin 1
        - logpdf(binom, tr[:world => :flips => 5], 2, 0.7) # removed last coin flip
        # increased number of flips but got same number heads:
        + sum(logpdf(binom, tr[:world => :flips => i], 3, 0.7) - logpdf(binom, tr[:world => :flips => i], 2, 0.7) for i=2:4)
    )
   
    @test isapprox(expected_score_delta, get_score(new_tr) - get_score(tr))
    @test isapprox(weight, expected_score_delta)
    @test retdiff != NoChange()
    @test all(new_tr[:world => :coin => i] == tr[:world => :coin => i] for i=2:4)
    @test discard == choicemap(
        (:world => :coin => 1 => :prob, 0.7),
        (:world => :coin => 5 => :prob, 0.7),
        (:world => :flips => 5 => :flips, tr[:world => :flips => 5])
    )
    @test all(get_retval(new_tr)[i] == get_retval(tr)[i] for i=1:4)
end