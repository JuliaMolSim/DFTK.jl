# High-level convenience functions to make standard models
# Note: When adding a function here, also add a method taking an AbstractSystem
#       to external/atomsbase.jl

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
