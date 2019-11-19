

"""
TODO DOCME
"""
function model_free_electron(lattice::AbstractMatrix, n_electrons; kwargs...)
    Model(lattice, n_electrons; kwargs...)
end

"""
TODO DOCME
"""
function model_hcore(lattice::AbstractMatrix, composition...; kwargs...)
    n_electrons = sum(length(pos) * n_elec_valence(spec) for (spec, pos) in composition)
    Model(lattice, n_electrons;
          external=term_external(composition...),
          nonlocal=term_nonlocal(composition...), kwargs...)
end

"""
TODO DOCME
"""
function model_dft(lattice::AbstractMatrix, functionals, composition...; kwargs...)
    n_electrons = sum(length(pos) * n_elec_valence(spec) for (spec, pos) in composition)
    Model(lattice, n_electrons;
          external=term_external(composition...),
          nonlocal=term_nonlocal(composition...),
          hartree=term_hartree(),
          xc=term_xc(functionals...), kwargs...)
end
function model_dft(lattice::AbstractMatrix, functionals::Symbol, composition...; kwargs...)
    model_dft(lattice, [functionals], composition...; kwargs...)
end

"""
TODO DOCME
"""
function model_reduced_hf(lattice::AbstractMatrix, composition...; kwargs...)
    n_electrons = sum(length(pos) * n_elec_valence(spec) for (spec, pos) in composition)
    Model(lattice, n_electrons;
          external=term_external(composition...),
          nonlocal=term_nonlocal(composition...),
          hartree=term_hartree())
end
