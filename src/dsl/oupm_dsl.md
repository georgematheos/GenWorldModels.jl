# The OUPM DSL
This is a DSL for writing OUPMs with number statements.

## Restrictions
- **Commands cannot be looped over.**  Any special command used in a generative function
  within a world model must only be called at most once in an execution of that generative
  function; this means that special commands *cannot* be used in loops or comprehensions.
  This is both due to implementation details, and because it is much more performant
  to use `@map_get` and `@setmap_get` than to perform many individual `@get` calls.
  (Commands like `@origin`, `@abstract`, etc., are equivalent to `@get origin[...]`, etc.,
  and thus can be looped over with eg. `@map_get origin[list_of_objects]`.)

### TODOs:
- Should we support non-property generative functions which get access to the world somehow,
    so they can access things like `@origin`, `@get`, etc.?