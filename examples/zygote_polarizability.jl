# # Polarizability using Zygote

using DFTK
using LinearAlgebra
using Zygote

## Construct PlaneWaveBasis given a particular electric field strength
## Again we take the example of a Helium atom.
He = ElementPsp(:He, psp=load_psp("hgh/lda/He-q2"))
atoms = [He => [[1/2; 1/2; 1/2]]]  # Helium at the center of the box
function make_basis(ε::T; a=10., Ecut=30) where T
    lattice=T(a) * I(3)  # lattice is a cube of ``a`` Bohrs
    # model = model_DFT(lattice, atoms, [:lda_x, :lda_c_vwn];
    #                   extra_terms=[ExternalFromReal(r -> -ε * (r[1] - a/2))],
    #                   symmetries=false)
    terms = [
        Kinetic(),
        AtomicLocal(),
        # AtomicNonlocal(),
        # Ewald(),
        # PspCorrection(),
        # Entropy(),
        # Hartree(),
        ExternalFromReal(r -> -ε * (r[1] - a/2))
    ]
    model = Model(lattice; atoms, terms, temperature=1e-3)
    PlaneWaveBasis(model, Ecut; kgrid=[1, 1, 1])  # No k-point sampling on isolated system
end

## dipole moment of a given density (assuming the current geometry)
function dipole(basis, ρ)
    # @assert isdiag(basis.model.lattice)
    a  = basis.model.lattice[1, 1]
    rr = [a * (r[1] - 1/2) for r in r_vectors(basis)]
    sum(rr .* ρ) * basis.dvol
end

## Function to compute the dipole for a given field strength
function compute_dipole(ε; tol=1e-8, kwargs...)
    scfres = self_consistent_field(make_basis(ε; kwargs...), tol=tol)
    dipole(scfres.basis, scfres.ρ)
end


# With this in place we can compute the polarizability from finite differences
# (just like in the previous example):
polarizability_fd = let
    ε = 0.01
    (compute_dipole(ε) - compute_dipole(0.0)) / ε
end
# 0.6198098991595362

Zygote.gradient(compute_dipole, 0.0) # TODO

# debug output:
# (size(ψ), size.(ψ)) = ((1,), [(7809, 1)])
# (size(∂ψ), size.(∂ψ)) = ((1,), [(7809, 8)])
# (size(occupation), size.(occupation)) = ((1,), [(1,)])
# after DFTK.select_occupied_orbitals:
# (size(∂ψ), size.(∂ψ)) = ((1,), [(7809, 1)])
# (size(∂Hψ), size.(∂Hψ)) = ((1,), [(7809, 1)])
# ∂H = ChainRulesCore.NoTangent()
# δenergies = ChainRulesCore.ZeroTangent()

# TODO next steps:
# find out why mul_pullback returns NoTangent here for ∂H
