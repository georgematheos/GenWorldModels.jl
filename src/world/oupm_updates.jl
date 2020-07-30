# struct IDAssociationChanged <: Gen.Diff end

# # returns an set of all indices which have had their associations changed 
# # and whose old IDs have had their associations changed
# function move_all_between!(world, type; min, inc, max=Inf)
#     new_id_table, changed_ids = move_all_between(world.id_table, oupm_type_name(type); min=min, inc=inc, max=max)
#     world.id_table = new_id_table
#     return changed_ids
# end

# function perform_oupm_move_and_enqueue_downstream!(world, spec)
#     table_before_update = world.id_table
#     changed_ids = perform_oupm_move!(world, spec)
#     for id in changed_ids
#         getindex_call = Call(_get_index_addr, spec.type(id))
#         world.state.diffs[getindex_call] = IDAssociationChanged()
#         enqueue_all_downstream_calls_and_note_dependency_has_diff!(world, getindex_call)

#         old_idx = get_idx(table_before_update, spec.type, id)
#         getid_call = Call(spec.type, old_idx)
#         world.state.diffs[getid_call] = IDAssociationChanged()
#         enqueue_all_downstream_calls_and_note_dependency_has_diff!(world, getid_call)

#         if has_id(world.id_table, spec.type, id)
#             new_idx = get_idx(world.id_table, spec.type, id)
#             if !has_idx(table_before_update, spec.type, new_idx)
#                 note_new_call!(world, Call(spec.type, new_idx))
#             end
#         end
#     end
# end

# function perform_oupm_move!(world, spec::BirthMove)
#     return move_all_between!(world, spec.type; min=spec.idx, inc=1)
# end
# function perform_oupm_move!(world, spec::DeathMove)
#     return move_all_between!(world, spec.type; min=spec.idx+1, inc=-1)
# end
# function perform_oupm_move!(world, spec::SplitMove)
#     from_idx = spec.from_idx
#     to_idx1 = min(spec.to_idx1, spec.to_idx2)
#     to_idx2 = max(spec.to_idx1, spec.to_idx2)

#     # capture the variables common to all the `move_all_between!` calls we need, to simplify syntax
#     changed_indices = Set{UUID}()
#     mb! = let w=world, c=changed_indices, t=spec.type
#         (min, max, inc) -> union!(c, move_all_between!(w, t; min=min, max=max, inc=inc))
#     end

#     if from_idx < to_idx1
#         mb!(from_idx + 1, to_idx1, -1),
#         mb!(to_idx2, Inf, +1)
#     elseif from_idx < to_idx2
#         mb!(to_idx1, from_idx - 1, +1),
#         mb!(to_idx2, Inf, +1)
#     else
#         mb!(from_idx + 1, Inf, +1),
#         mb!(to_idx2 - 1, from_idx - 1, +2),
#         mb!(to_idx1, to_idx2 - 2, +1)
#     end

#     return changed_indices
# end
# function perform_oupm_move!(world, spec::MergeMove)
#     to_idx = spec.to_idx
#     from_idx1 = min(spec.from_idx1, spec.from_idx2)
#     from_idx2 = max(spec.from_idx1, spec.from_idx2)

#     # capture the variables common to all the `move_all_between!` calls we need, to simplify syntax
#     changed_indices = Set{UUID}()
#     mb! = let w=world, c=changed_indices, t=spec.type
#         (min, max, inc) -> union!(c, move_all_between!(w, t; min=min, max=max, inc=inc))
#     end

#     if to_idx < from_idx1
#         mb!(from_idx2 + 1, Inf, -1),
#         mb!(to_idx, from_idx1 - 1, +1)
#     elseif to_idx < from_idx2
#         mb!(from_idx2 + 1, Inf, -1),
#         mb!(from_idx1 + 1, to_idx, -1)
#     else
#         mb!(from_idx1 + 1, from_idx2 - 1, -1),
#         mb!(from_idx2 + 1, to_idx + 1, -2),
#         mb!(to_idx + 2, Inf, -1)
#     end

#     return changed_indices
# end
# function perform_oupm_move!(world, spec::MoveMove)
#     typename = oupm_type_name(spec.type)
    
#     _has_id = has_idx(world.id_table, typename, spec.from_idx)

#     if _has_id
#         id = get_id(world.id_table, spec.type, spec.from_idx)
#     end
#     if spec.from_idx < spec.to_idx
#         changed_ids = move_all_between!(world, spec.type; min=spec.from_idx+1, max=spec.to_idx, inc=-1)
#     else
#         changed_ids = move_all_between!(world, spec.type; min=spec.to_idx, max=spec.from_idx-1, inc=+1)
#     end
#     if _has_id
#         world.id_table = insert_id(world.id_table, typename, spec.to_idx, id)
#         push!(changed_ids, id)
#     end

#     return changed_ids
# end

# function reverse_moves(moves::Tuple)
#     Tuple(reverse_move(moves[j]) for j=length(moves):-1:1)
# end
# reverse_move(m::BirthMove) = DeathMove(m.type, m.idx)
# reverse_move(m::DeathMove) = BirthMove(m.type, m.idx)
# reverse_move(m::SplitMove) = MergeMove(m.type, m.from_idx, m.to_idx1, m.to_idx2)
# reverse_move(m::MergeMove) = SplitMove(m.type, m.to_idx, m.from_idx1, m.from_idx2)
# reverse_move(m::MoveMove) = MoveMove(m.type, m.to_idx, m.from_idx)