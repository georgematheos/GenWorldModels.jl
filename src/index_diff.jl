### Interface ###

"""
    IndexDiff
A diff representing that the value an object is storing at certain
indices have changed, and this is the only way in which the object
has changed.
"""
abstract type IndexDiff <: Gen.Diff end

"""
    get_diff_for_index(idxdiff::IndexDiff, index)
    
Returns the diff for `index` in `idxdiff`.
"""
function get_diff_for_index(idxdiff::IndexDiff, index)
    error("Not Implemented")
end
Base.getindex(idxdiff::IndexDiff, index) = get_diff_for_index(idxdiff, index)

"""
    has_diff_for_index(idxdiff::IndexDiff, index)
    
Returns true if there is a diff for `index` in `idxdiff`, false otherwise.
"""
function has_diff_for_index(idxdiff::IndexDiff, index)
    get_diff_for_index(idxdiff, index) != NoChange()
end

### Concrete IndexDiffs ###

"""
    MutableIndexDiff
    
An `IndexDiff` which can store any number of indices' diffs
and which can dynamically add or modify diffs.
"""
struct MutableIndexDiff <: IndexDiff
    diffs::Dict{<:Any, Diff}
end
function MutableIndexDiff(idx_to_diff::Pair{Any, <:Diff}...)
    MutableIndexDiff(Dict(idx_to_diff...))
end

"""
    add_diff!(idxdiff::MutableIndexDiff{T}, idx::T, diff::Diff) where {T}
Adds `diff` to `idxdiff` for index `idx`, or overwrites the value currently at `idx`
with `diff`.
"""
function add_diff!(idxdiff::MutableIndexDiff, idx, diff::Diff)
    idxdiff.diffs[idx] = diff
end

has_diff_for_index(diff::MutableIndexDiff, idx) = haskey(diff.diffs, idx) && diff.diffs[idx] != NoChange()
get_diff_for_index(diff::MutableIndexDiff, idx) = get(diff.diffs, idx, NoChange())