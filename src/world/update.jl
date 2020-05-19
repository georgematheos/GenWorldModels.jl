
struct UpdateWorldState <: WorldState
    constraints
end
struct RegenerateWorldState <: WorldState
    constraints
end

function lookup_or_generate_during_update!(world, call)

end