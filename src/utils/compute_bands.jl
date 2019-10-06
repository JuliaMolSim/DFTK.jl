
# TODO Think about this function some more
#      Better interface would be to integrate closer with pymatgen
#      and actually offer an interface taking the kline_density instead of the kpoint coordinates ...

"""
TODO docme
"""
function compute_bands(ham::Hamiltonian, n_bands::Integer, kcoords; kwargs...)
    kpoints = build_kpoints(ham.basis, kcoords)
    lobpcg(ham, n_bands, kpoints; interpolate_kpoints=false, kwargs...)
end
