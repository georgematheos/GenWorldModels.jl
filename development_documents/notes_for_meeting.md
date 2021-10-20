Possible issues/bugs:

- Possible integer overflow for calls.


Performance:

- Update algorithm asymptotically fast but allocates many data structures
  - Could have a faster update spec guaranteeing no order change
    - Could we automatically detect this?
    - Could we "bake" the traces, as Marco calls it, to "memoize" the update sequence until
    we update the topological order again


Questions:
- Efficiency of automatically deleting unused calls vs explicitely proposing to remove them?
- Are there any important use cases where we actually need such flexible tracking of topological orders, etc.?
  - Does this answer change if there is vs isn't a significant performance penalty for this?




Args splatted for mgfs?