module Tests
using Test
using Gen
using GenWorldModels

import SpecialFunctions: logbeta, loggamma
logbeta(x::Vector{<:Real}) = sum(loggamma(i) for i in x) - loggamma(sum(x))
using Distributions: BetaBinomial
using StatsBase: sample

include("../types.jl")
include("../model2/model2.jl")

@gen function _foo(ar, α, β)
    num ~ uniform_discrete(1, 2)
    subsets ~ Map(beta_bernoulli_subset)(fill(ar, num), fill(α, num), fill(β, num))
    return subsets
end
foo = UsingWorld(_foo)

@gen function proposal(tr)
    if tr[:kernel => :num] == 1
        set = tr[:kernel => :subsets][1]
        for item in set
            if {:handle => item} ~ bernoulli(0.5)
                {:do_with => item} ~ uniform_discrete(1, 3)
            end
        end
    end
end

@oupm_involution (old, fwd) to (new, bwd) begin
    if @read(old[:kernel => :num], :disc) == 1
        @write(new[:kernel => :num], 2, :disc)
        subset = @read(old[:kernel => :subsets], :disc)[1]
        for item in subset
            if @read(fwd, :handle => item, :disc)
                dowith = @read(fwd, :do_with => item, :disc)
                if dowith == 1
                    @write(bwd)
                elseif dowith == 2

                else

                end
            else
                @save_for_reverse_regenerate(:kernel => :subsets => 1 => item)
                @regenerate(:kernel => :subsets => 1 => item)
                @regenerate(:kernel => :subsets => 2 => item)
            end
        end
        @regenerate(:kernel => :subsets => 1)
        @regenerate(:kernel => :subsets => 2)
    else
        @write(new[:kernel => :num], 1, :disc)
        
    end
end