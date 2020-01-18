

"""
TODO DOCME
"""
function model_free_electron(lattice::AbstractMatrix, n_electrons; kwargs...)
    Model(lattice, n_electrons; kwargs...)
end

"""
TODO DOCME
"""
function model_hcore(lattice::AbstractMatrix, atoms; kwargs...)
    n_electrons = sum(length(pos) * n_elec_valence(spec) for (spec, pos) in atoms)
    Model(lattice, n_electrons;
          external=term_external(atoms),
          nonlocal=term_nonlocal(atoms), kwargs...)
end

"""
TODO DOCME
"""
function model_dft(lattice::AbstractMatrix, functionals, atoms; kwargs...)
    n_electrons = sum(length(pos) * n_elec_valence(spec) for (spec, pos) in atoms)
    Model(lattice, n_electrons;
          external=term_external(atoms),
          nonlocal=term_nonlocal(atoms),
          hartree=term_hartree(),
          xc=term_xc(functionals...), kwargs...)
end
function model_dft(lattice::AbstractMatrix, functionals::Symbol, atoms; kwargs...)
    model_dft(lattice, [functionals], atoms; kwargs...)
end

"""
TODO DOCME
"""
function model_reduced_hf(lattice::AbstractMatrix, atoms; kwargs...)
    n_electrons = sum(length(pos) * n_elec_valence(spec) for (spec, pos) in atoms)
    Model(lattice, n_electrons;
          external=term_external(atoms),
          nonlocal=term_nonlocal(atoms),
          hartree=term_hartree(), kwargs...)
end
