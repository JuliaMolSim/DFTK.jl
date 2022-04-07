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

    pspkey = list_psp(:Ar, functional="lda")[1].identifier
    particles = [
        Atom(:Ar, [21.0, 21.0, 21.0]u"bohr"; pseudopotential=pspkey),
        Atom(:Ar, [7.0, 21.0, 21.0]u"bohr"; pseudopotential=pspkey),
        Atom(:Ar, [21.0, 7.0, 21.0]u"bohr"; pseudopotential=pspkey),
        Atom(:Ar, [7.0, 7.0, 21.0]u"bohr"; pseudopotential=pspkey),
        Atom(:Ar, [21.0, 21.0, 7.0]u"bohr"; pseudopotential=pspkey),
        Atom(:Ar, [7.0, 21.0, 7.0]u"bohr"; pseudopotential=pspkey),
        Atom(:Ar, [21.0, 7.0, 7.0]u"bohr"; pseudopotential=pspkey),
        Atom(:Ar, [7.0, 7.0, 7.0]u"bohr"; pseudopotential=pspkey)
    ]
    box = [[28.0, 0.0, 0.0], [0.0, 28.0, 0.0], [0.0, 0.0, 28.0]]u"bohr"
    boundary_conditions = [Periodic(), Periodic(), Periodic()]
    system = FlexibleSystem(particles, box, boundary_conditions)

    eandf = energy_and_force(system, potential)
    @test eandf.e isa Float64
    @test eandf.f isa AbstractVector{<:SVector{3,<:Float64}}

    @test haskey(potential.scf_kwargs, :ψ)
    @test haskey(potential.scf_kwargs, :ρ)
end
