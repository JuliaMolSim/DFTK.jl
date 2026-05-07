# Quick test script for TRS k-point reduction
# Run as: julia --project test_trs.jl

using Revise
using DFTK
using LinearAlgebra

println("=== Testing SymOp with θ field ===")

# Test basic SymOp construction
s1 = SymOp(Mat3{Int}(I), Vec3(0.0, 0.0, 0.0))
s2 = SymOp(Mat3{Int}(I), Vec3(0.0, 0.0, 0.0); θ=-1)
@assert s1.θ == 1 "Default θ should be +1"
@assert s2.θ == -1 "Explicit θ=-1"
@assert isone(s1) "Identity should be isone"
@assert !isone(s2) "θ=-1 identity is not isone"
@assert s1 != s2 "θ=+1 ≠ θ=-1"
println("  SymOp construction: OK")

# Test group composition
s3 = s1 * s2
@assert s3.θ == -1 "(+1)×(-1) = -1"
s4 = s2 * s2
@assert s4.θ == 1 "(-1)×(-1) = +1"
println("  Group composition: OK")

# Test inverse
@assert inv(s1).θ == 1 "inv of unitary is unitary"
@assert inv(s2).θ == -1 "inv of antiunitary is antiunitary"
println("  Inverse: OK")

println()
println("=== Testing Si SCF with TRS symmetry ===")

a = 10.263141334305642  # Si lattice constant in Bohr
lattice = a / 2 * [[0 1 1]; [1 0 1]; [1 1 0]]
Si = ElementPsp(:Si, load_psp("hgh/lda/Si-q4"))
atoms = [Si, Si]
positions = [ones(3)/8, -ones(3)/8]

# No-symmetry run
model_nosym = model_LDA(lattice, atoms, positions; symmetries=false)
basis_nosym = PlaneWaveBasis(model_nosym; Ecut=10, kgrid=[4, 4, 4])
println("  No-symmetry k-points: $(length(basis_nosym.kpoints))")

# Default (with TRS) run
model_trs = model_LDA(lattice, atoms, positions)
basis_trs = PlaneWaveBasis(model_trs; Ecut=10, kgrid=[4, 4, 4])
n_trs_symops = count(s -> s.θ == -1, model_trs.symmetries)
println("  TRS symmetry k-points: $(length(basis_trs.kpoints))")
println("  Model TRS symops count: $n_trs_symops")
println("  k-weight sum: $(sum(basis_trs.kweights))")

@assert abs(sum(basis_trs.kweights) - 1.0) < 1e-10 "k-weights must sum to 1"

# Check k-count is less with TRS
@assert length(basis_trs.kpoints) <= length(basis_nosym.kpoints) "TRS should not increase k-count"
println("  k-point counts: nosym=$(length(basis_nosym.kpoints)), TRS=$(length(basis_trs.kpoints))")

println()
println("=== Running SCF with TRS symmetry ===")

scfres_nosym = self_consistent_field(basis_nosym; tol=1e-8)
scfres_trs   = self_consistent_field(basis_trs;   tol=1e-8)

E_nosym = scfres_nosym.energies.total
E_trs   = scfres_trs.energies.total
println("  Energy (no-sym): $E_nosym")
println("  Energy (TRS):    $E_trs")
println("  Difference:      $(abs(E_nosym - E_trs))")
@assert abs(E_nosym - E_trs) < 1e-6 "Energies should agree"

println()
println("All tests passed!")
