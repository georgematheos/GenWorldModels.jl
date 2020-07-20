"""
    LookupCounts

An immutable datatype to track the dependencies in the world and the number of times each lookup has occurred.
"""
struct LookupCounts
    lookup_counts::PersistentHashMap{Call, Int} # counts[call] is the number of times this was looked up
    # dependency_counts[call1][call2] = # times call1 looked up in call2
    # ie. indexing is dependency_counts[looked_up, looker_upper]
    dependency_counts::PersistentHashMap{Call, PersistentHashMap{Call, Int}}
end
LookupCounts() = LookupCounts(PersistentHashMap{Call, Int}(), PersistentHashMap{Call, PersistentHashMap{Call, Int}}())

function note_new_call(lc::LookupCounts, call::Call)
    new_lookup_counts = assoc(lc.lookup_counts, call, 0)
    new_dependency_counts = assoc(lc.dependency_counts, call, PersistentHashMap{Call, Int}())
    LookupCounts(new_lookup_counts, new_dependency_counts)
end

function note_new_lookup(lc::LookupCounts, call::Call, call_stack::Stack{Call})
    new_lookup_counts = assoc(lc.lookup_counts, call, lc.lookup_counts[call] + 1)
    if !isempty(call_stack)
        looked_up_in = first(call_stack)

        if haskey(lc.dependency_counts[call], looked_up_in)
            new_count = lc.dependency_counts[call][looked_up_in] + 1
        else
            new_count = 1
        end

        new_dependency_counts = assoc(lc.dependency_counts, call,
            assoc(lc.dependency_counts[call], looked_up_in, new_count)
        )
    else
        new_dependency_counts = lc.dependency_counts
    end
    LookupCounts(new_lookup_counts, new_dependency_counts)
end
# note lookup removal from another call (ie. not from kernel)
function note_lookup_removed(lc::LookupCounts, call::Call, called_from::Call)
    new_count = lc.dependency_counts[call][called_from] - 1
    if new_count == 0
        new_dependency_counts = assoc(lc.dependency_counts, call,
            dissoc(lc.dependency_counts[call], called_from)
        )
    else
        new_dependency_counts = assoc(lc.dependency_counts, call,
        assoc(lc.dependency_counts[call], called_from, new_count)
        )
    end

    new_count = lc.lookup_counts[call] - 1
    if new_count == 0 && is_mgf_call(call)
        # if we now have 0 lookups, and this is a mgf call (rather than a world arg),
        # remove this key from the lookup_counts dictionary
        new_lookup_counts = dissoc(lc.lookup_counts, call)
        new_dependency_counts = dissoc(new_dependency_counts, call)
    else
        new_lookup_counts = assoc(lc.lookup_counts, call, new_count)
    end

    (LookupCounts(new_lookup_counts, new_dependency_counts), new_count)
end
# note lookup removed from kernel (ie. not removed from another call)
function note_lookup_removed(lc::LookupCounts, call::Call)
    new_count = lc.lookup_counts[call] - 1
    if new_count == 0
        if is_mgf_call(call)
            new_lookup_counts = dissoc(lc.lookup_counts, call)
            new_dependency_counts = dissoc(lc.dependency_counts, call)
        end
    else
        new_lookup_counts = assoc(lc.lookup_counts, call, new_count)
        new_dependency_counts = lc.dependency_counts
    end

    (LookupCounts(new_lookup_counts, new_dependency_counts), new_count)
end

function get_all_calls_which_look_up(lc::LookupCounts, call)
    (call_which_looked_up for (call_which_looked_up, count) in lc.dependency_counts[call] if count > 0)
end

"""
    get_number_of_expected_kernel_lookups(lc::LookupCounts, call::Call)

Gives the number of traced lookups to `call` which must have occurred in the kernel
for the lookup counts object to have the number of counts it has tracked,
assuming all other calls have been properly tracked.
"""
function get_number_of_expected_kernel_lookups(lc::LookupCounts, call::Call)
    count = lc.lookup_counts[call]
    for (looker_upper, dependency_lookup_count) in lc.dependency_counts[call]
        count -= dependency_lookup_count
    end
    return count
end
