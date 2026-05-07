# Quick test script for TRS k-point reduction
# Run as: julia test_trs.jl

using Revise
using DFTK
using LinearAlgebra

println("=== Testing SymOp with θ field ===")

s1 = SymOp(Mat3{Int}(I), Vec3(0.0, 0.0, 0.0))
s2 = SymOp(Mat3{Int}(I), Vec3(0.0, 0.0, 0.0); θ=-1)
@assert s1.θ == 1 "Default θ should be +1"
@assert s2.θ == -1 "Explicit θ=-1"
@assert isone(s1) "Identity should be isone"
@assert !isone(s2) "θ=-1 identity is not isone"
@assert s1 != s2 "θ=+1 ≠ θ=-1"
s3 = s1 * s2; @assert s3.θ == -1 "(+1)×(-1) = -1"
s4 = s2 * s2; @assert s4.θ == 1  "(-1)×(-1) = +1"
@assert inv(s1).θ == 1  "inv of unitary is unitary"
@assert inv(s2).θ == -1 "inv of antiunitary is antiunitary"
println("  OK")

# -----------------------------------------------------------------------
# Helper: run SCF + forces, return (scfres, forces)
# -----------------------------------------------------------------------
function run_system(lattice, atoms, positions; use_symmetries)
    model = model_LDA(lattice, atoms, positions; symmetries=use_symmetries)
    basis = PlaneWaveBasis(model; Ecut=10, kgrid=[4, 4, 4])
    scfres = self_consistent_field(basis; tol=1e-10, callback=identity)
    forces = compute_forces_cart(scfres)
    scfres, forces
end

function check_results(scf_nosym, f_nosym, scf_trs, f_trs; tol_E=1e-6, tol_f=1e-4, tol_ρ=1e-6)
    ΔE = abs(scf_nosym.energies.total - scf_trs.energies.total)
    Δf = maximum(norm.(f_nosym .- f_trs))
    Δρ = maximum(abs.(scf_nosym.ρ .- scf_trs.ρ))
    println("  ΔE = $ΔE,  max|Δf| = $Δf,  max|Δρ| = $Δρ")
    @assert ΔE < tol_E  "Energies should agree (got $ΔE)"
    @assert Δf < tol_f  "Forces should agree (got $Δf)"
    @assert Δρ < tol_ρ  "Densities should agree (got $Δρ)"
end

Ga = ElementPsp(:Ga, load_psp("hgh/lda/Ga-q3"))
As = ElementPsp(:As, load_psp("hgh/lda/As-q5"))
a_gaas = 10.68   # Bohr
lattice_gaas = a_gaas / 2 * [[0 1 1]; [1 0 1]; [1 1 0]]

# -----------------------------------------------------------------------
# Test 1: GaAs zinc-blende (equilibrium) — Td spatial + TRS
# Forces are zero (equilibrium geometry).
# -----------------------------------------------------------------------
println("\n=== Test 1: GaAs zinc-blende (equilibrium) ===")

pos_gaas = [[0, 0, 0], [1/4, 1/4, 1/4]]

scf_nosym1, f_nosym1 = run_system(lattice_gaas, [Ga, As], pos_gaas; use_symmetries=false)
scf_trs1,   f_trs1   = run_system(lattice_gaas, [Ga, As], pos_gaas; use_symmetries=true)

n_sp1 = count(s -> s.θ == 1, scf_trs1.basis.model.symmetries) - 1
n_tr1 = count(s -> s.θ == -1, scf_trs1.basis.model.symmetries)
println("  Spatial symops: $n_sp1,  TRS symops: $n_tr1")
println("  k-points: nosym=$(length(scf_nosym1.basis.kpoints)), TRS=$(length(scf_trs1.basis.kpoints))")
@assert n_sp1 == 23  "GaAs Td has 23 non-identity spatial symops"
@assert n_tr1 == 24  "GaAs has 24 TRS symops"
@assert abs(sum(scf_trs1.basis.kweights) - 1.0) < 1e-10
check_results(scf_nosym1, f_nosym1, scf_trs1, f_trs1)
println("  OK")

# -----------------------------------------------------------------------
# Test 2: rattled GaAs — breaks all spatial symmetries, only TRS remains.
# Non-zero forces: exercises the full symmetrize_ρ + force path with TRS.
# Two different species → no generalized inversion even after rattling.
# -----------------------------------------------------------------------
println("\n=== Test 2: rattled GaAs (TRS-only, non-zero forces) ===")

δ = 0.04
pos_rattle = [[0, 0, 0] .+ δ * [0.3, -0.2, 0.5],
              [1/4, 1/4, 1/4] .+ δ * [-0.4, 0.1, -0.3]]

scf_nosym2, f_nosym2 = run_system(lattice_gaas, [Ga, As], pos_rattle; use_symmetries=false)
scf_trs2,   f_trs2   = run_system(lattice_gaas, [Ga, As], pos_rattle; use_symmetries=true)

n_sp2 = count(s -> s.θ == 1, scf_trs2.basis.model.symmetries) - 1
n_tr2 = count(s -> s.θ == -1, scf_trs2.basis.model.symmetries)
println("  Spatial symops: $n_sp2,  TRS symops: $n_tr2")
println("  k-points: nosym=$(length(scf_nosym2.basis.kpoints)), TRS=$(length(scf_trs2.basis.kpoints))")
@assert n_sp2 == 0 "Rattled GaAs should have no spatial symmetries"
@assert n_tr2 == 1 "Rattled GaAs should have 1 TRS symop"
# With TRS only, TRIM points (2k≡0 mod lattice) aren't halved; for [4,4,4] there
# are 8 such points so the irreducible count is (64+8)/2 = 36, not 64/2 = 32.
@assert length(scf_trs2.basis.kpoints) < length(scf_nosym2.basis.kpoints) "TRS should reduce k-count"
@assert length(scf_trs2.basis.kpoints) == 36 "Expected 36 irreducible k-points with TRS only on [4,4,4]"
@assert abs(sum(scf_trs2.basis.kweights) - 1.0) < 1e-10

# Sanity check: forces are genuinely non-zero
f_mag = maximum(norm.(f_nosym2))
println("  Max |f| (nosym) = $f_mag  [should be >> 0]")
@assert f_mag > 0.01 "Forces should be non-zero for rattled geometry"

check_results(scf_nosym2, f_nosym2, scf_trs2, f_trs2)
println("  OK")

println("\nAll tests passed!")
