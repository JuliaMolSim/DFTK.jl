# # Polarizability using automatic differentiation
#
# Simple example for computing properties using (forward-mode)
# automatic differentation.
# For a more classical approach and more details about computing polarizabilities,
# see [Polarizability by linear response](@ref).

using DFTK
using Zygote
using LinearAlgebra
using ForwardDiff

## Construct PlaneWaveBasis given a particular electric field strength
## Again we take the example of a Helium atom.

He = ElementPsp(:He, psp=load_psp("hgh/lda/He-q2"))
atoms = [He]
positions = [[1/2; 1/2; 1/2]] # Helium at the center of the box

function make_basis(ε::T; a=10., Ecut=30) where T
    lattice=T(a) * Mat3(I(3))  # lattice is a cube of ``a`` Bohrs

    model = model_DFT(lattice, atoms, positions, [:lda_x, :lda_c_vwn];
                      extra_terms=[ExternalFromReal(r -> -ε * (r[1] - a/2))],
                      symmetries=false)
    PlaneWaveBasis(model; Ecut, kgrid=[1, 1, 1])  # No k-point sampling on isolated system
end

function make_basis_model(ε::T; a=10., Ecut=30) where T
    lattice=T(a) * Mat3(I(3))  # lattice is a cube of ``a`` Bohrs
    terms = [
        Kinetic(),
        AtomicLocal(),
        AtomicNonlocal(),
        Ewald(),
        PspCorrection(),
        ExternalFromReal(r -> -ε * (r[1] - a/2)),
        Xc([:lda_x, :lda_c_vwn])
    ]
    model = Model(lattice, atoms, positions; terms, symmetries=false)
    PlaneWaveBasis(model; Ecut, kgrid=[1, 1, 1])  # No k-point sampling on isolated system
end

## dipole moment of a given density (assuming the current geometry)
function dipole(basis, ρ)
    #@assert isdiag(basis.model.lattice)
    a  = basis.model.lattice[1, 1]
    rr = [a * (r[1] - 1/2) for r in r_vectors(basis)]
    sum(rr .* ρ) * basis.dvol
end

## Function to compute the dipole for a given field strength
function compute_dipole(ε; tol=1e-8, kwargs...)
    scfres = self_consistent_field(make_basis_model(ε; kwargs...), tol=tol)
    dipole(scfres.basis, scfres.ρ)
end;

# With this in place we can compute the polarizability from finite differences
# (just like in the previous example):
polarizability_fd = let
    ε = 0.01
    (compute_dipole(ε) - compute_dipole(0.0)) / ε
end

f = compute_dipole(0.0)
fd = polarizability_fd
g = Zygote.gradient(compute_dipole,0.0)

println("f ", f, " fd ", fd, " g ", g)
