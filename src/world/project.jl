function project_(world::World, selection::Selection)
    weight = 0.
    for (call, trace) in world.subtraces
        weight += project(trace, selection[addr(call) => key(call)])
    end
    return weight
end