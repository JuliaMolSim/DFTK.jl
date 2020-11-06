include("operators.jl")

### Terms
# - A Term is something that, given a state, returns a named tuple (E, hams) with an energy
#   and a list of RFO (for each kpoint).
# - Each term must overload
#     `ene_ops(t, ψ, occ; kwargs...)` -> (E::Real, ops::Vector{RealFourierOperator}).
# - Note that terms are allowed to hold on to references to ψ (eg Fock term),
#   so ψ should not mutated after ene_ops

# The Hamiltonian is defined as half of the gradient of the energy
# with respect to the density matrix sum_n fn |ψn><ψn|.
# In particular, dE/dψn = 2 fn |Hψn> (plus weighting for kpoint sampling)
abstract type Term end

### Builders are objects X that store the term parameters, and produce a
# XTerm <: Term when instantiated with a `basis`



"""
A term with a constant zero energy.
"""
struct NoopTerm <: Term
    basis::PlaneWaveBasis
end
function ene_ops(term::NoopTerm, ψ, occ; kwargs...)
    (E=zero(eltype(term.basis)), ops=[NoopOperator(term.basis, kpoint)
                                      for kpoint in term.basis.kpoints])
end

include("Hamiltonian.jl")

# breaks_symmetries on a term builder answers true if this term breaks
# the symmetries of the lattice/atoms (in which case kpoint reduction
# is invalid)
breaks_symmetries(term_builder::Any) = false

include("kinetic.jl")

include("local.jl")
breaks_symmetries(term_builder::ExternalFromReal) = true
breaks_symmetries(term_builder::ExternalFromFourier) = true

include("nonlocal.jl")
include("hartree.jl")
include("power_nonlinearity.jl")
include("xc.jl")
include("ewald.jl")
include("psp_correction.jl")
include("entropy.jl")

include("magnetic.jl")
breaks_symmetries(term_builder::Magnetic) = true

include("anyonic.jl")
breaks_symmetries(term_builder::Anyonic) = true


# forces computes either nothing or an array forces[el][at][α]
compute_forces(term::Term, ψ, occ; kwargs...) = nothing  # by default, no force

"""
Compute the forces of an obtained SCF solution. Returns the forces wrt. the fractional
lattice vectors. To get cartesian forces use [`compute_forces_cart`](@ref).
Returns a list of lists of forces
`[[force for atom in positions] for (element, positions) in atoms]`
which has the same structure as the `atoms` object passed to the underlying [`Model`](@ref).
"""
@timing function compute_forces(basis::PlaneWaveBasis, ψ, occ; kwargs...)
    if !any(iszero(kpt.coordinate) for kpt in basis.kpoints)
        @warn "Forces for shifted k-Grids not tested"
    end

    # TODO optimize allocs here
    T = eltype(basis)
    forces = [zeros(Vec3{T}, length(positions)) for (element, positions) in basis.model.atoms]
    for term in basis.terms
        f_term = compute_forces(term, ψ, occ; kwargs...)
        if !isnothing(f_term)
            forces += f_term
        end
    end
    forces
end

"""
Compute the cartesian forces of an obtained SCF solution in Hartree / Bohr.
Returns a list of lists of forces
`[[force for atom in positions] for (element, positions) in atoms]`
which has the same structure as the `atoms` object passed to the underlying [`Model`](@ref).
"""
function compute_forces_cart(basis::PlaneWaveBasis, ψ, occ; kwargs...)
    lattice = basis.model.lattice
    forces = compute_forces(basis::PlaneWaveBasis, ψ, occ; kwargs...)
    [[lattice \ f for f in forces_for_element] for forces_for_element in forces]
end

function compute_forces(scfres)
    compute_forces(scfres.basis, scfres.ψ, scfres.occupation; ρ=scfres.ρ, ρspin=scfres.ρspin)
end
function compute_forces_cart(scfres)
    compute_forces_cart(scfres.basis, scfres.ψ, scfres.occupation;
                        ρ=scfres.ρ, ρspin=scfres.ρspin)
end


@doc raw"""
    compute_kernel(basis::PlaneWaveBasis; kwargs...)

Computes a matrix representation of the full response kernel
(derivative of potential with respect to density) in real space.
For non-spin-polarized calculations the matrix dimension is
`prod(basis.fft_size)` × `prod(basis.fft_size)` and
for collinear spin-polarized cases it is
`2prod(basis.fft_size)` × `2prod(basis.fft_size)`.
In this case the matrix has effectively 4 blocks, which are:
```math
\left(\begin{array}{cc}
    K_{α, \text{tot}} & K_{α, \text{spin}}\\
    K_{β, \text{tot}} & K_{β, \text{spin}}
\end{array}\right)
```
i.e. corresponding to a mapping
``(ρ_\text{tot}, ρ_\text{spin})^T = (ρ_α + ρ_β, ρ_α - ρ_β)^T ↦ (V_α, V_β)^T``.
"""
@timing function compute_kernel(basis::PlaneWaveBasis{T}; kwargs...) where {T}
    n_spin = basis.model.n_spin_components
    kernel = zeros(T, n_spin * prod(basis.fft_size), n_spin * prod(basis.fft_size))
    for term in basis.terms
        isnothing(term) && continue
        kernel .+= compute_kernel(term; kwargs...)
    end
    kernel
end
compute_kernel(::Term; kwargs...) = nothing  # By default no kernel


"""
    apply_kernel(basis::PlaneWaveBasis, dρ, dρspin=nothing; kwargs...)

Computes the potential response to a perturbation `(dρ, dρspin)` in real space.
Returns the array `[dV_α, dV_β]` for collinear spin-polarized
calculations, else the array [dV_{tot}].
"""
@timing function apply_kernel(basis::PlaneWaveBasis, dρ, dρspin=nothing;
                              RPA=false, kwargs...)
    n_spin = basis.model.n_spin_components
    (n_spin == 1) && @assert isnothing(dρspin)
    (n_spin == 2) && @assert !isnothing(dρspin)
    @assert 1 ≤ n_spin ≤ 2

    dV = [RealFourierArray(basis) for _ in 1:n_spin]
    for term in basis.terms
        # Skip XC term if RPA is selected
        RPA && term isa TermXc && continue

        dV_term = apply_kernel(term, dρ, dρspin; kwargs...)
        if !isnothing(dV_term)
            for σ in 1:n_spin
                dV[σ].real .+= dV_term[σ].real
            end
        end
    end
    dV
end
apply_kernel(::Term, dρ, dρspin; kwargs...) = nothing  # by default, no kernel
