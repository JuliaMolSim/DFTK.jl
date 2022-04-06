using AtomsBase
using DFTK
using InteratomicPotentials
using StaticArrays
using Test
using Unitful
using UnitfulAtomic

@testset "DFTK -> InteratomicPotentials" begin
    potential1 = DFTKPotential(5, [1, 1, 1]; damping=0.7, tol=1e-4)
    potential2 = DFTKPotential(136.05693123044495u"eV", [1, 1, 1]; damping=0.7, tol=1e-4)

    @test potential1.Ecut ≈ potential2.Ecut

    pspkey = list_psp(:Ar, functional="lda")[1].identifier
    particles = [AtomsBase.Atom(:Ar, [i & 1 == 0 ? 7 : 21, i & 2 == 0 ? 7 : 21, i & 4 == 0 ? 7 : 21]u"bohr"; pseudopotential=pspkey) for i ∈ 0:7]
    box = [[28.0, 0, 0], [0, 28.0, 0], [0, 0, 28.0]]u"bohr"
    boundary_conditions = [Periodic(), Periodic(), Periodic()]
    system = FlexibleSystem(particles, box, boundary_conditions)

    eandf = energy_and_force(system, potential1)
    @test eandf.e isa Float64
    @test eandf.f isa AbstractVector{<:SVector{3,<:Float64}}

    @test potential1.previous_scf[].energies.total == eandf.e
    @test compute_forces_cart(potential1.previous_scf[]) == eandf.f
end
