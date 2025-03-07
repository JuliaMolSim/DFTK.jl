# test force using finite differences
#
using AtomsBuilder
using AtomsIO
using Unitful
using UnitfulAtomic
using JLD2
using DFTK
using PseudoPotentialData
using Test
setup_threading(; n_blas=1)

function test_forces(system; kgrid, Ecut, symmetries=true, ε=1e-5, atol=1e-8)
    function compute_energy(dx, ε)
        atomsmod = map(system, position(system, :) + ε * dx) do atom, pos
            Atom(species(atom), pos)
        end
        sysmod = FlexibleSystem(system; atoms=atomsmod)

        pseudopotentials = PseudoFamily("dojo.nc.sr.pbe.v0_4_1.standard.upf")
        pseudopotentials = PseudoFamily("cp2k.nc.sr.pbe.v0_1.semicore.gth")
        model = model_DFT(sysmod; functionals=PBE(), pseudopotentials, symmetries)
        model = model_atomic(sysmod; pseudopotentials, symmetries, temperature=1e-3)

        terms = [Kinetic(), AtomicLocal(), PspCorrection(), Entropy(),
                 Ewald(), AtomicNonlocal(),  ]
        model = Model(sysmod; model_name="custom", terms, pseudopotentials, symmetries, temperature=1e-3)
        basis = PlaneWaveBasis(model; kgrid, Ecut)

        self_consistent_field(basis; tol=1e-12)
    end

    for i in 1:1 #length(system)
        dx = [zeros(3) * u"Å" for _ in 1:length(system)]
        dx[i]  = 0.1randn((3, )) * u"Å"
        dx[i]  = [0.1, 0.02, 0.1] * u"Å"

        scfres = compute_energy(dx, 0.0)
        forces = compute_forces_cart(scfres)
        Fε_ref = sum(map(forces, dx) do Fi, dxi
            -dot(Fi, austrip.(dxi))
        end)

        Fε = let
            (  compute_energy(dx,  ε).energies.total
             - compute_energy(dx, -ε).energies.total) / 2ε
        end

        @show Fε maximum(abs, Fε_ref - Fε)
        @test maximum(abs, Fε_ref - Fε) < atol
    end
end

#=
@testset "Silicon" begin
    system = bulk(:Si)
    rattle!(system, 1e-3u"Å")
    test_forces(system, kgrid=[2, 2, 2], Ecut=30, ε=1e-5, atol=1e-8)
end
=#

@testset "Rutile 1" begin
    system = load_system("SnO2(1).cif")
    rattle!(system, 1e-1u"Å")
    for ε in (1e-2, 1e-3, 1e-4, 1e-5)
        test_forces(system; kgrid=[1, 1, 1], Ecut=20, ε, atol=1e-8)
    end
end

@testset "Rutile 2" begin
    system = load_system("geo3.extxyz")
    test_forces(system; kgrid=[2, 2, 3], Ecut=30, ε=1e-5, atol=1e-7)
end

#=
@testset "Rutile" begin
    system = load_system("geo2.extxyz")
    test_forces(system; kgrid=[6, 6, 9], Ecut=40,
                        symmetries=false)
end
=#
