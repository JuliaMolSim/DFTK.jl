using AtomsBase
using DFTK
using InteratomicPotentials
using StaticArrays
using Test
using Unitful
using UnitfulAtomic

@testset "DFTK -> InteratomicPotentials" begin
    potential = DFTKPotential(5u"hartree", [1, 1, 1]; functionals=[:lda_x, :lda_c_pw], scf_kwargs=Dict(:damping => 0.7, :tol => 1e-4))

    pspkey = list_psp(:Ar, functional="lda")[1].identifier
    particles = [AtomsBase.Atom(:Ar, [i & 1 == 0 ? 7 : 21, i & 2 == 0 ? 7 : 21, i & 4 == 0 ? 7 : 21]u"bohr"; pseudopotential=pspkey) for i ∈ 0:7]
    box = [[28.0, 0, 0], [0, 28.0, 0], [0, 0, 28.0]]u"bohr"
    boundary_conditions = [Periodic(), Periodic(), Periodic()]
    system = FlexibleSystem(particles, box, boundary_conditions)

    eandf = energy_and_force(system, potential)
    @test eandf.e isa Float64
    @test eandf.f isa AbstractVector{<:SVector{3,<:Float64}}

    @test haskey(potential.scf_kwargs, :ψ)
    @test haskey(potential.scf_kwargs, :ρ)
end
