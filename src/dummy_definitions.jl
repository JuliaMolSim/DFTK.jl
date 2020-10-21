"""
    load_scfres(filename)

Load back an `scfres`, which has previously been stored with `save_scfres`.
Note the warning in `save_scfres`.
"""
function load_scfres end



"""
    save_scfres(filename, scfres)

Save an `scfres` obtained from `self_consistent_field` to a JLD2 file.

!!! warning "No compatibility guarantees"
    No guarantees are made with respect to this function at this point.
    It may change incompatibly between DFTK versions or stop working / be removed
    in the future.
"""
function save_scfres end
