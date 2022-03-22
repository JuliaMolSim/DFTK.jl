# # Polarizability using Zygote

using DFTK
using LinearAlgebra
using Zygote
using Logging
using Random

Random.seed!(0)

Ne = ElementPsp(:Ne, psp=load_psp("hgh/lda/Ne-q8"))
atoms = [Ne]
positions = [[1/2; 1/2; 1/2]]
function make_basis(ε::T; a=10., Ecut=5) where T
    lattice=T(a) * Mat3(I(3))  # lattice is a cube of ``a`` Bohrs
    terms = [
        Kinetic(),
        AtomicLocal(),
        AtomicNonlocal(),
        ExternalFromReal(r -> -ε * (r[1] - a/2)),
    ]
    model = Model(lattice, atoms, positions; terms, symmetries=false)
    PlaneWaveBasis(model; Ecut, kgrid=[1, 1, 1])  # No k-point sampling on isolated system
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
    ε = 0.001
    (compute_dipole(ε) - compute_dipole(0.0)) / ε
end

f = compute_dipole(0.0)
g = Zygote.gradient(compute_dipole, 0.0)
println("f: ", f, " fd: ",polarizability_fd, " AD: ",g)
