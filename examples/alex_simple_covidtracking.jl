using Gen
using GeoJSON
using PolygonOps
using Geodesy
using LightGraphs
using GraphPlot
import JSON

########################
#  LOAD MIT GEOMETRY   #
########################

const MIT_BUILDINGS = GeoJSON.parsefile("mit-geo.json").features
Base.zero(::Type{Vector{Float64}}) = [0., 0.]
buildings = [(building.properties["facility"], PolygonOps.centroid(building.geometry.coordinates[1][1])) for building in MIT_BUILDINGS]
origin_lla = LLA(buildings[70][2][2], buildings[70][2][1], 0.0) # Building 46
buildings_lla = [(b[1], LLA(b[2][2], b[2][1], 0.0)) for b in buildings]
bcs_centric_frame = ENUfromLLA(buildings_lla[70][2], wgs84)
buildings_enu = [(b[1], bcs_centric_frame(b[2])) for b in buildings_lla]
back_to_lla = inv(bcs_centric_frame)

struct PointOfInterest
  name :: String
  enu :: ENU{Float64}
end

# MIT dorms:
home_building_names = ["62", "64", "W51", "W7", "W1", "W61", "W4", "W70",
                       "W71", "W46", "NW61", "W79", "E2", "NW35", "NW10"]
pois = map(x -> PointOfInterest(x[1], x[2]), filter(x -> !isnothing(x[1]), buildings_enu))
homes = filter(x -> in(x.name, home_building_names), pois)
work_building_names = collect(filter(x -> !in(x, home_building_names),
                              map(x -> x[1], buildings_enu)))

works = filter(x -> in(x.name, work_building_names), pois)



#############
#  Config   #
#############
struct SimulationConfig
  home_locations :: Vector{PointOfInterest}
  work_locations :: Vector{PointOfInterest}
  num_agents :: Int
  duration :: Float64 # in hours
  source_rate :: Int  # Poisson rate for number of sources total
  # Agent, Test time
  tests :: Vector{Tuple{Int, Float64}}
end

@dist choose_proportionally(probs, l) = l[categorical(probs)]
@dist uniform_from_list(l) = l[uniform_discrete(1, length(l))]

#############
#  Agents   #
#############

# Agents have a home location and a work location.
@gen function generate_agent(config)
  home ~ uniform_from_list(config.home_locations)
  work ~ uniform_from_list(config.work_locations)
  return (home = home, work = work)
end

#############
#  Sources  #
#############

# Simplified source model: each source is near some base location, and has
# no other parameters.
@gen function generate_source(config)
  base ~ uniform_from_list([config.home_locations; config.work_locations])
  e ~ normal(base.enu.e, 40)
  n ~ normal(base.enu.n, 40)
  return ENU(e, n, 0.0)
end


##############
#  Movement  #
##############

# Waiting time distribution
@dist next_time(current_time, rate) = current_time + exponential(rate)

# Basic walking speed
METERS_PER_HOUR = 5000

# Simulate agent location, if they were last simulated at location `from`,
# have the current goal of getting to `to`, and `dt` hours have elapsed since
# we last simulated the location.
@gen function simulate_location(from, dt, to)
  # Very crude -- replace!
  distance_traveled = dt * METERS_PER_HOUR
  distance_to_goal = distance(from, to)
  fraction_covered = min(1.0, distance_traveled / distance_to_goal)
  e_loc = from.e + fraction_covered * (to.e - from.e)
  n_loc = from.n + fraction_covered * (to.n - from.n)
  e ~ normal(e_loc, 20 + abs(e_loc - from.e) / 2.0)
  n ~ normal(n_loc, 20 + abs(n_loc - from.n) / 2.0)
  ENU(e, n, 0.0)
end


@gen function generate_motion_path(agent, modeled_times, config)
  # Current time
  t = 0.0
  # Lowest index such that t < modeled_times[obs_idx].
  modeled_time_index = 1

  # Simulated times and locations
  times = Float64[]
  locs  = ENU[]

  # Most recent goal and location (TODO: don't just initialize to home)
  last_goal        = agent.home.enu
  last_sampled_loc = agent.home.enu

  # Keep track of which of the sampled locations will later be observed
  # (just the indices into the main locs/times arrays)
  to_observe = []

  # How many times have we changed goal locations (+ 1)?
  i = 1

  # Main loop
  while t < config.duration
    # Poisson rate = 0.25 goal changes per hour
    next_t = {(:event_times, i)} ~ next_time(t, 0.25)

    # We now need to sample any "true locations" between t and next_t,
    # as well as sampling a "true location" at next_t.
    while modeled_time_index <= length(modeled_times) && next_t > modeled_times[modeled_time_index]

      # Interpolate (ish) between last location and the previously held goal
      last_sampled_loc = {(:location, modeled_times[modeled_time_index])} ~ simulate_location(last_sampled_loc, modeled_times[modeled_time_index] - t, last_goal)

      # Update t, now that new location has been simulated
      t = modeled_times[modeled_time_index]

      # Bookkeeping
      push!(times, t)
      push!(locs, last_sampled_loc)
      push!(to_observe, length(locs))
      modeled_time_index += 1
    end

    # Even if there were no observations, sample a current location before changing goals
    last_sampled_loc = {(:location, next_t)} ~ simulate_location(last_sampled_loc, next_t - t, last_goal)
    t = next_t
    push!(times, t)
    push!(locs, last_sampled_loc)

    # Now that we're caught up, the agent chooses a new goal location.
    # Could be anywhere, but work + home are preferred
    possible_buildings = [agent.home, agent.work, config.home_locations..., config.work_locations...]
    building_probs = [0.4, 0.4, [0.2/(length(possible_buildings)-2) for i=1:(length(possible_buildings)-2)]...]
    last_goal = ({(:goals, i)} ~ choose_proportionally(building_probs, possible_buildings)).enu

    # Increment number of goal changes.
    i += 1
  end

  return times, locs, to_observe
end

# For contact graph estimation, we need a deterministic way of interpolating
# between sampled points. Here's a simple one, which performs linear interpolation,
# except that if the gap between A and B is long enough (temporally), it assumes
# that the agent quickly got from A to B and has mostly been sitting around at B.
function interpolate_loc(t1, loc1, t2, loc2, t)
  time_from_a_to_b = distance(loc1, loc2) * METERS_PER_HOUR
  if time_from_a_to_b >= t2 - t1
    frac = (t - t1) / (t2 - t1)
    e = loc1.e + (loc2.e - loc1.e) * frac
    n = loc1.n + (loc2.n - loc1.n) * frac
    return ENU(e, n, 0.0)
  end

  if t - t1 < time_from_a_to_b
    frac = (t - t1) / (time_from_a_to_b - t1)
    e = loc1.e + (loc2.e - loc1.e) * frac
    n = loc1.n + (loc2.n - loc1.n) * frac
    return ENU(e, n, 0.0)
  end

  return loc2
end




###################
#  Contact Graph  #
###################

# Two types of event: agent was near a source, or
# was near another agent.

abstract type ContactGraphNode end
struct SourceAgentNode <: ContactGraphNode
  source :: Int
  agent  :: Int
  t1 :: Float64
  t2 :: Float64
end
struct AgentAgentNode <: ContactGraphNode
  i  :: Int
  j  :: Int
  t1 :: Float64
  t2 :: Float64
end

# Messy, repetitive function to compute the contact graph.
# Takes in all generated sources, as well as arrays times and
# locs, where, e.g., locs[i] is itself an array of Agent i's locations.
@gen function compute_contact_graph(sources, times, locs, config)
  # We're going to discretize time for the purpose of the graph.
  # Loop through in 15-minute increments, and check if two people
  # are in the same place at that time.

  vertices = []
  event_graph = DiGraph() # DiGraph = "Directed Graph"

  num_agents = length(times)

  # Keeps track, for each agent, of *highest* index idx such that
  # times[agent_id][idx] < t (the current time).
  agent_timestep_indices = ones(Int, length(locs))

  # Keeps track of when interactions began (in discretized time).
  # Keys are either (i, j) (for agent i and agent j) or (:source, i, j)
  # (for agent i and source j). If an entry exists, it will be a timestamp,
  # and indicates that from that time until time `t`, the two entities in question
  # have been in contact. (When they go out of contact, we remove the dictionary
  # entry and add a node to the graph representing the entire interval.)
  intersection_start_times = Dict()

  # Loop through in increments of 15 minutes.
  for t in 0:0.25:config.duration
    # Compute agent locations at this time, interpolating between the two
    # closest measurements we have.
    agent_locations = []
    for i in 1:num_agents
      current_index = agent_timestep_indices[i]
      if current_index < length(times[i]) && times[i][current_index + 1] < t
        agent_timestep_indices[i] = (current_index += 1)
      end
      push!(agent_locations, interpolate_loc(times[i][current_index], locs[i][current_index], times[i][current_index+1], locs[i][current_index+1], t))
    end

    # Perform pairwise "close enough to infect" tests between agents.
    for i=1:num_agents
      for j=i:num_agents
        if distance(agent_locations[i], agent_locations[j]) < 40
          # Interaction is happening -- check if this is a new interaction.
          if !haskey(intersection_start_times, (i, j))
            intersection_start_times[(i, j)] = t
          end
        else
          # No interaction -- if we *had* been tracking an interaction,
          # remove it from our tracker and add it to the contact graph.
          if haskey(intersection_start_times, (i, j))
            # Intersection ended; add to graph
            start = intersection_start_times[(i, j)]
            add_vertex!(event_graph)
            push!(vertices, AgentAgentNode(i, j, start, t))
            delete!(intersection_start_times, (i, j))
          end
        end
      end
    end

    # Agent/Source interactions
    for i=1:num_agents
      for (j, s) in enumerate(sources)
        if distance(s, agent_locations[i]) < 40
          # Interaction is happening; start tracking if we're not yet.
          if !haskey(intersection_start_times, (:source, i, j))
            intersection_start_times[(:source, i, j)] = t
          end
        else
          # No interaction; if we *had* been tracking an interaction, remove it
          # and add as a graph node.
          if haskey(intersection_start_times, (:source, i, j))
            start = intersection_start_times[(:source, i, j)]
            add_vertex!(event_graph)
            push!(vertices, SourceAgentNode(j, i, start, t))
            delete!(intersection_start_times, (:source, i, j))
          end
        end
      end
    end
  end

  # Now add edges, from a node *i* to the two nodes representing the nexrt
  # interactions in which each of the participants at node *i* take place.
  # Compute this by looping through all nodes in order of start time.
  ordering = sortperm(vertices, by = x -> x.t1)
  # Keeps track of the node number that each agent was last involved with.
  # (If an agent doesn't appear, they haven't been in any previous nodes.)
  last_node_involving = Dict()
  for idx in ordering
    v = vertices[idx]
    if v isa SourceAgentNode
      if haskey(last_node_involving, v.agent)
        add_edge!(event_graph, last_node_involving[v.agent], idx)
      end
      last_node_involving[v.agent] = idx
    elseif v isa AgentAgentNode
      if haskey(last_node_involving, v.i)
        add_edge!(event_graph, last_node_involving[v.i], idx)
      end
      if haskey(last_node_involving, v.j)
        add_edge!(event_graph, last_node_involving[v.j], idx)
      end
      last_node_involving[v.i] = idx
      last_node_involving[v.j] = idx
    end
  end
  vertices, event_graph
end



###################
#  Transmissions  #
###################

import Distributions

# Process contact graph in topological order to simulate transitions.
# At Source/Agent nodes, simulate for whether the agent is infected.
# (This depends on whether agent was already infected, on base rate
# [including duration since we last simulated status of this agent], and
# on duration of encounter with source.)
# At Agent/Agent nodes, simulate for whether each agent has been infected
# since we last saw them, and then, if exactly 1 of the two agents is infected,
# for whether the infection is transmitted.
@gen function simulate_transmissions(event_graph, vertices, config)
  # Health status will either be :s (susceptible) or (:i, t) for "infected at time t".
  health_status = Any[:s for i=1:config.num_agents]
  # We assume that we have "seen" each agent at time 0; because we assume that
  # health_status[agent] is always correct at the time last_seen_time[agent], this
  # means we assume that at time 0, all agents are healthy.
  last_seen_time = zeros(config.num_agents)

  # Topological sort on contact graph
  order_to_process = topological_sort_by_dfs(event_graph)

  for node in order_to_process
    v = vertices[node]


    if v isa SourceAgentNode
      # How long has it been since we last simulated for this agent?
      elapsed = v.t1 - last_seen_time[v.agent]
      # Are they already infected?
      already_infected = health_status[v.agent] == :s ? 0.0 : 1.0
      # Assume a base rate of ~1 infection per person every 3,000,000 hours.
      prob_of_exogenous_infection = Distributions.cdf(Distributions.Exponential(3e6), elapsed)
      # Assume an increased rate near the source, of ~1 infection per person per 3 hours spent with the source.
      prob_of_source_infection = Distributions.cdf(Distributions.Exponential(3), v.t2-v.t1)
      # Is v.agent infected by the end of this interaction?
      infected = {(:source_infections, v.source, v.agent, v.t1)} ~ bernoulli(max(prob_of_exogenous_infection + prob_of_source_infection, already_infected))
      if infected && iszero(already_infected)
        health_status[v.agent] = (:i, v.t1)
      end
      last_seen_time[v.agent] = v.t2
    end

    if v isa AgentAgentNode
      # Agent i
      i_elapsed = v.t1 - last_seen_time[v.i]
      already_infected = health_status[v.i] == :s ? 0.0 : 1.0
      prob_of_exogenous_infection = Distributions.cdf(Distributions.Exponential(3e6), i_elapsed)

      # Include v.j in the address to distinguish between multiple
      # v.i nodes that start at the same time.
      i_infected = {(:base_infections, v.i, v.t1, v.j)} ~ bernoulli(max(already_infected, prob_of_exogenous_infection))
      if i_infected && iszero(already_infected)
        health_status[v.i] = (:i, v.t1)
      end
      last_seen_time[v.i] = v.t2

      # Agent j
      j_elapsed = v.t1 - last_seen_time[v.j]
      already_infected = health_status[v.j] == :s ? 0.0 : 1.0
      prob_of_exogenous_infection = Distributions.cdf(Distributions.Exponential(3e6), j_elapsed)
      j_infected = {(:base_infections, v.j, v.t1, v.i)} ~ bernoulli(max(already_infected, prob_of_exogenous_infection))
      if j_infected && iszero(already_infected)
        health_status[v.j] = (:i, v.t1)
      end
      last_seen_time[v.j] = v.t2

      # Transmission
      if xor(i_infected, j_infected)
        prob_of_transmission = Distributions.cdf(Distributions.Exponential(3), v.t2-v.t1)
        transmitted = {(:transmissions, v.i, v.j, v.t1)} ~ bernoulli(prob_of_transmission)
        if transmitted
          if i_infected
            health_status[v.j] = (:i, v.t1)
          else
            health_status[v.i] = (:i, v.t1)
          end
        end
      end
    end
  end

  return health_status
end




###################
#  Observations   #
###################

@gen function generate_observations(times, true_locations)
  [(t, ENU({(t, :e)} ~ normal(l.e, 1.0),   {(t, :n)} ~ normal(l.n, 1.0))) for (t, l) in zip(times, true_locations)]
end

@gen function generate_testing_data(testing_times, health_status)
  # For now, just 95% accurate, no matter when the disease was contracted.
  tests = []
  for (agent, t) in testing_times
    is_infected = health_status[agent] != :s && health_status[agent][2] < t
    prob_pos = is_infected ? 0.95 : 0.05
    push!(tests, {(agent, t)} ~ bernoulli(prob_pos))
  end
  tests
end




###################
#  Full Model     #
###################


@gen function campus_model(config)
  # Generate agents
  agents = [{:agents => i} ~ generate_agent(config) for i=1:config.num_agents]

  # Generate motion and observations
  motion_times = []
  motion_locs  = []
  for i=1:config.num_agents
    n_modeled_times = {:modeled_times => i => :n} ~ poisson(100)
    modeled_times =  [{:modeled_times => i =>  j} ~ uniform(0, config.duration) for j=1:n_modeled_times]
    times, locs, to_observe = {:motions => i} ~ generate_motion_path(agents[i], sort(modeled_times), config)
    push!(motion_times, times)
    push!(motion_locs,  locs)
    {:observed_locations => i} ~ generate_observations(times[to_observe], locs[to_observe])
  end

  # Generate sources
  num_sources ~ poisson(config.source_rate)
  sources = [{:sources => i} ~ generate_source(config) for i=1:num_sources]

  # Compute contact graph
  vertices, contact_graph = {:graph} ~ compute_contact_graph(sources, motion_times, motion_locs, config)

  # Simulate along contact graph
  transmissions ~ simulate_transmissions(contact_graph, vertices, config)

  # Simulate tests
  tests ~ generate_testing_data(config.tests, transmissions)

  return tests, transmissions
end

# JSON from existing stuff
# Modeled locations vs. observed locations.
# 50 agents, 2 distinguished agents -- positions specified as hard-wired lists -- fine-grained (1000) and coarse-grained (50, plus 1000 modeled).
#     virtual agents, let's have poisson(30) modeled locations.
#  observation_source categorical variable -- fine-grained and more trustworthy, coarse-grained and less trust-worthy.
#  sample trajectories including goals and goal times, and

loc_dict(loc) = begin
  ll = back_to_lla(loc)
  Dict(:lat => ll.lat, :lon => ll.lon)
end
function get_goals(tr, agent_num)
  i = 1
  goals = []
  while has_value(get_choices(tr), :motions => agent_num => (:goals, i))
    push!(goals, Dict(:time => tr[:motions => agent_num => (:event_times, i)],
                      :loc  => loc_dict(tr[:motions => agent_num => (:goals, i)].enu)))
    i += 1
  end
  goals
end

function vertex_json(tr, i)
  verts, g = tr[:graph]
  vert = verts[i]

  if vert isa AgentAgentNode
    agent1 = Dict(:agent => vert.i, :infected => tr[:transmissions => (:base_infections, vert.i, vert.t1, vert.j)])
    agent2 = Dict(:agent => vert.j, :infected => tr[:transmissions => (:base_infections, vert.j, vert.t1, vert.i)])
    transmitted = xor(agent1[:infected], agent2[:infected]) ? tr[:transmissions => (:transmissions, vert.i, vert.j, vert.t1)] : false
    Dict(:transmitted => transmitted,
         :agent1 => agent1, :agent2 => agent2, :t1 => vert.t1, :t2 => vert.t2)
  elseif vert isa SourceAgentNode
    agent = Dict(:agent => vert.agent, :infected => tr[:transmissions => (:source_infections, vert.source, vert.agent, vert.t1)])
    source = vert.source
    Dict(:agent => agent, :source => source, :t1 => vert.t1, :t2 => vert.t2)
  end
end


function trace_to_dict(t)
  cfg = get_args(t)[1]
  d = Dict()

  # Fill with all the trace data.

  # Sources:
  d[:num_sources] = tr[:num_sources]
  source_locations = [tr[:sources => i] for i in 1:d[:num_sources]]
  d[:sources] = [loc_dict(l) for l in source_locations]

  # Agents:
  # Each agent has a home and work
  # as well as times and locations
  # and goals
  agent_dicts = []
  for i=1:cfg.num_agents
    goals = get_goals(t, i)
    times, locs = tr[:motions => i]
    locs = [Dict(:time => time, :loc => loc_dict(loc)) for (time, loc) in zip(times, locs)]
    observations = map(x -> Dict(:time => x[1], :loc => loc_dict(x[2])), tr[:observed_locations => i])
    push!(agent_dicts,
                 Dict(:goals => goals, :locs => locs,
                      :observed => observations,
                      :home => loc_dict(tr[:agents => i].home.enu),
                      :work => loc_dict(tr[:agents => i].work.enu)))

  end
  d[:agents] = agent_dicts

  # Now I need the contact graph and transmission data.
  verts, g = tr[:graph]
  d[:contacts] = [vertex_json(tr, i) for i in 1:length(verts)]
  d[:infection_times] = [h == :s ? nothing : h[2] for h in tr[:transmissions]]

  # And finally, testing.
  d[:tests] = [Dict(:result => tr[:tests][i], :agent => test[1], :time => test[2]) for (i, test) in enumerate(cfg.tests)]

  return d
end

# Smart initialization based on trajectories.
@dist poisson_but_at_least(n, rate) = n + poisson(max(1, rate-n))
singleton(x) = [x]
@dist exactly(x) = singleton(x)[categorical([1.0])]
function initialize_with_trajectories(config, known_trajectories)
  @gen function smart_initializer(config, known_trajectories)
    for (i, traj) in enumerate(known_trajectories)
      traj = [(t, bcs_centric_frame(loc)) for (t, loc) in traj]
      n = {:modeled_times => i => :n} ~ poisson_but_at_least(length(known_trajectories), 100)
      for (j, (t, loc)) in enumerate(traj)
        {:modeled_times => i => j} ~ exactly(t)
        {:motions => i => (:location, t) => :e} ~ normal(loc.e, 1.0)
        {:motions => i => (:location, t) => :n} ~ normal(loc.n, 1.0)
        {:observed_locations => i => (t, :e)} ~ exactly(loc.e)
        {:observed_locations => i => (t, :n)} ~ exactly(loc.n)
      end

      # Now choose goal points intelligently-ish.
      t = 0.0
      event_num = 1
      while t < config.duration
        # Poisson rate = 0.25 goal changes per hour
        t = {:motions => i => (:event_times, event_num)} ~ next_time(t, 0.25)
        all_possible_locations = [config.home_locations..., config.work_locations...]
        closest_observed_time_index = argmin([abs(t - pt[1]) for pt in traj])
        goal_index = argmin([distance(l.enu, traj[closest_observed_time_index][2]) for l in all_possible_locations])
        goal = all_possible_locations[goal_index]
        {:motions => i => (:goals, event_num)} ~ exactly(goal)
        event_num += 1
      end
    end
  end
  constraints = get_choices(simulate(smart_initializer, (config, known_trajectories)))
  display(constraints)
  tr, = Gen.generate(campus_model, (config,), constraints)
  tr
end

buildings_lla[88]

config = (SimulationConfig(homes, works, 50, 48.0, 5, [(i, 25.0) for i=1:10]))
tr = initialize_with_trajectories(config, [[(0.3, LLA(42.3623391, -71.0916622, 0.0)), (1.2, LLA(42.361569, -71.090521879, 0.0))]])
#tr = simulate(campus_model, (config,)))
println(JSON.json(trace_to_dict(tr)))