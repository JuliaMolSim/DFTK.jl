# Convenience functions to make standard models

"""
Convenience constructor, which builds a standard atomic (kinetic + atomic potential) model.
Use `extra_terms` to add additional terms.
"""
function model_atomic(lattice::AbstractMatrix, atoms::Vector, positions::Vector;
                      extra_terms=[], kwargs...)
    @assert !(:terms in keys(kwargs))
    terms = [Kinetic(),
             AtomicLocal(),
             AtomicNonlocal(),
             Ewald(),
             PspCorrection(),
             extra_terms...]
    if :temperature in keys(kwargs) && kwargs[:temperature] != 0
        terms = [terms..., Entropy()]
    end
    Model(lattice, atoms, positions; model_name="atomic", terms, kwargs...)
end


"""
Build a DFT model from the specified atoms, with the specified functionals.
"""
function model_DFT(lattice::AbstractMatrix, atoms::Vector, positions::Vector, xc::Xc;
                   extra_terms=[], kwargs...)
    model_name = isempty(xc.functionals) ? "rHF" : join(xc.functionals, "+")
    model_atomic(lattice, atoms, positions;
                 extra_terms=[Hartree(), xc, extra_terms...], model_name, kwargs...)
end
function model_DFT(lattice::AbstractMatrix, atoms::Vector, positions::Vector, functionals;
                   kwargs...)
    model_DFT(lattice, atoms, positions, Xc(functionals); kwargs...)
end


"""
Build an LDA model (Teter93 parametrization) from the specified atoms.
"""
function model_LDA(lattice::AbstractMatrix, atoms::Vector, positions::Vector; kwargs...)
    model_DFT(lattice, atoms, positions, [:lda_x, :lda_c_pw]; kwargs...)
end


"""
Build an PBE-GGA model from the specified atoms.
DOI:10.1103/PhysRevLett.77.3865
"""
function model_PBE(lattice::AbstractMatrix, atoms::Vector, positions::Vector; kwargs...)
    model_DFT(lattice, atoms, positions, [:gga_x_pbe, :gga_c_pbe]; kwargs...)
end


"""
Build a SCAN meta-GGA model from the specified atoms.
DOI:10.1103/PhysRevLett.115.036402
"""
function model_SCAN(lattice::AbstractMatrix, atoms::Vector, positions::Vector; kwargs...)
    model_DFT(lattice, atoms, positions, [:mgga_x_scan, :mgga_c_scan]; kwargs...)
end


# NOTE:  This is a temporary function, which could disappear any time.
function atoms_compat(oldatoms::Vector{<:Pair})
    atoms = Element[]
    positions = Vec3[]
    for (at, poses) in oldatoms
        for pos in poses
            push!(atoms, at)
            push!(positions, pos)
        end
    end
    (atoms, positions)
end
@deprecate model_atomic(lattice, atoms::Vector{<:Pair}; kwargs...) model_atomic(lattice, DFTK.atoms_compat(atoms)...; kwargs...)
@deprecate model_DFT(lattice, atoms::Vector{<:Pair}, functionals; kwargs...) model_DFT(lattice, DFTK.atoms_compat(atoms)..., functionals; kwargs...)
@deprecate model_LDA(lattice, atoms::Vector{<:Pair}; kwargs...)  model_LDA(lattice,  DFTK.atoms_compat(atoms)...; kwargs...)
@deprecate model_PBE(lattice, atoms::Vector{<:Pair}; kwargs...)  model_PBE(lattice,  DFTK.atoms_compat(atoms)...; kwargs...)
@deprecate model_SCAN(lattice, atoms::Vector{<:Pair}; kwargs...) model_SCAN(lattice, DFTK.atoms_compat(atoms)...; kwargs...)
