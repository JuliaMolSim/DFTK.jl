# Developer's style guide

The following conventions are not strictly enforced in DFTK. The rule of thumb is
readability over consistency.

## Coding and formatting conventions

- Line lengths should be 92 characters.
- Avoid shortening variable names only for its own sake.
- Named tuples should be explicit, i.e., `(; var=val)` over `(var=val)`.
- Use NamedTuple unpacking to prevent ambiguities when getting multiple arguments from
  a function.
- Empty callbacks should use `identity`.
- Use `=` to loop over a range but `in` to loop over elements.
- Always format the `where` keyword with explicit braces: `where {T <: Any}`.
- Do not use implicit arguments in functions but explicit keyword.
- Prefix function name by `_` if it is an internal helper function, which is not of general
  use and should thus be kept close to the vicinity of the calling function.
