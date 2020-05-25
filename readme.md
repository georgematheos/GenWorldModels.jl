# Gen UsingWorld combinator

This repository implements a `UsingWorld` combinator for [Gen](gen.dev).
This combinator makes it easier to write open-universe Gen models with
memoization and lazy evaluation in a way which supports asymptotically efficient
MCMC.

### Use
The `UsingWorld` wraps a "kernel" generative function, and provides
the kernel access to a sort of state which can track values for calls
to memoized generative functions.  We call this state the "world".

As an example of usage, say we are tracking some aircrafts, and each
for each one, we take a measurement at each timestep t=1, 2, ..., up to infinity.
Of course, we can't actually generate infinity values, but we can write a generative function
which can generate the measurement for any time, as follows.


```julia
include("src/WorldModels.jl")
using .WorldModels

@gen function take_measurement(world, (aircraft_index, time))
    position ~ lookup_or_generate(world[:positions][(aircraft_index, time)])
    measured ~ normal(position, 0.5)
    return measured
end

@gen function generate_position(world, (aircraft_index, time))
    if time == 1
        position ~ position_prior()
    else
        prev_position ~ lookup_or_generate(world[:positions][(aircraft_index, time - 1)])
        position ~ aircraft_movement_model(prev_position)
    end
    return position
end

@gen function measure_aircraft_at_time(world, aircraft_index, time)
    measurement ~ lookup_or_generate(world[:measurements][(aircraft_index, time)])
    return measurement
end
measure_aircrafts_at_times = Map(measure_aircraft_at_time)

@gen function kernel(world, time_to_measure_at)
    num_aircrafts ~ poisson(5)
    measurements ~ measure_aircrafts_at_times(fill(world, num_aircrafts), collect(1:num_aircrafts), fill(time_to_measure_at, num_aircrafts))
    return measurements
end

get_measurements_at_time = UsingWorld(kernel, :measurement => take_measurement, :position => generate_position)

# example usage:
measurements_at_time_5 = get_measurements_at_time(5)
```
Here, `generate_position` and `take_measurement` are "memoized generative functions" within the context of 
the `UsingWorld` instance `get_measurements_at_time`.  In the construction of the `UsingWorld` instance,
these are given the addresses `:measurement` and `:position`.

The kernel for this generative function takes the `world`
as it's first argument.  This is the persistent state in which calls to the memoized generative functions
have their values recorded.  Another way to think about it is that
whenever we call memoized generative function, the world is populated with this
call.

All memoized generative functions must take 2 arguments: the `world`, and the argument to the MGF.
In the examples above, this argument is a tuple containing 2 values: `(aircraft_index, time)`.
Memoized generative functions may call other memoized generative functions, or call themselves
with different arguments.  This allows for recursion, as is used in `generate_position`.

To call a memoized generative function, one must use a TRACED call
```julia
{address} ~ lookup_or_generate(world[:memoized_gen_fn_address][argument_for_memoized_gen_fn])
```
This `lookup_or_generate` function will either call the memoized generative function on the given argument,
or if the world object is already populated with a value for this argument to the memoized generative function,
it will simply return this.  The `UsingWorld` combinator exposes no choicemap for `lookup_or_generate`, since this
function is deterministic given a world which has the needed value within it.

The choicemap exposed by `UsingWorld` has 2 top-level addresses: `:kernel` and `:world`.
The `:kernel` submap is simply the choicemap from the kernel.
The `:world` has a submap at address `:memoized_gen_function_address => arg` for every
call to a memoized generative function which was made in the execution which resulted in the trace; this
submap is simply the choicemap for the trace created when executing the memoized generative function on this argument.

To give an example, a trace for calling `get_measurements_at_time(3)` might look like
```
│
├── :kernel
│   │
│   ├── :num_aircrafts : 3
│
└── :world
    │
    ├── :measurements
    │   │
    │   ├── (2, 3)
    │   │   │
    │   │   ├── :measured : 0.6585363086020086
    │   │
    │   ├── (1, 3)
    │   │   │
    │   │   ├── :measured : 0.5815384425838848
    │   │
    │   └── (3, 3)
    │       │
    │       ├── :measured : 0.021430153402649477
    │
    └── :positions
        │
        ├── (2, 1)
        │
        ├── (1, 3)
        │   │
        │   ├── :position
        │   │   │
        │   │   └── :delta_x : 0.244975342277369
        │
        ├── (3, 1)
        │
        ├── (1, 2)
        │   │
        │   ├── :position
        │   │   │
        │   │   └── :delta_x : 0.0006385394142925336
        │
        ├── (2, 2)
        │   │
        │   ├── :position
        │   │   │
        │   │   └── :delta_x : 0.0244428260187015
        │
        ├── (2, 3)
        │   │
        │   ├── :position
        │   │   │
        │   │   └── :delta_x : 0.5447047828556035
        │
        ├── (1, 1)
        │
        ├── (3, 3)
        │   │
        │   ├── :position
        │   │   │
        │   │   └── :delta_x : 0.015559778670828785
        │
        └── (3, 2)
            │
            ├── :position
            │   │
            │   └── :delta_x : 0.304434668479045
```
(if `delta_x` is a choice exposed in the `aircraft_movement_model`).

To see a working implementation of this model, look at `test/aircrafts.jl`.