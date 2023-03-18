using AtomsBase
using DFTK
using InteratomicPotentials
using StaticArrays
using Test
using Unitful
using UnitfulAtomic

@testset "DFTK -> InteratomicPotentials" begin
    functionals = [:lda_x, :lda_c_pw]
    scf_kwargs = Dict(:damping => 0.7, :tol => 1e-4)
    potential = DFTKPotential(5u"hartree", [1, 1, 1]; functionals, scf_kwargs)

    particles = [
        :Ar => [21.0, 21.0, 21.0]u"bohr",
        :Ar => [7.0, 21.0, 21.0]u"bohr",
        :Ar => [21.0, 7.0, 21.0]u"bohr",
        :Ar => [7.0, 7.0, 21.0]u"bohr",
        :Ar => [21.0, 21.0, 7.0]u"bohr",
        :Ar => [7.0, 21.0, 7.0]u"bohr",
        :Ar => [21.0, 7.0, 7.0]u"bohr",
        :Ar => [7.0, 7.0, 7.0]u"bohr"
    ]
    box = [[28.0, 0.0, 0.0], [0.0, 28.0, 0.0], [0.0, 0.0, 28.0]]u"bohr"
    system = attach_psp(periodic_system(particles, box); Ar="hgh/lda/ar-q8.hgh")

    eandf = energy_and_force(system, potential)
    @test eandf.e isa Unitful.Energy
    @test eandf.f isa AbstractVector{<:SVector{3,<:Unitful.Force}}

    @test haskey(potential.scf_kwargs, :ψ)
    @test haskey(potential.scf_kwargs, :ρ)
end
