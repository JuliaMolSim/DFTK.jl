using DFTK
using Unitful
using UnitfulAtomic
using AtomsBase
using Test
import DFTK: compute_pspmap

@testset "DFTK -> AbstractSystem -> DFTK" begin
    Si = ElementCoulomb(:Si)
    C  = ElementPsp(:C, psp=load_psp("hgh/pbe/c-q4.hgh"))
    H  = ElementPsp(:H, psp=load_psp("hgh/lda/h-q1.hgh"))

    lattice   = randn(3, 3)
    atoms     = [Si, C, H, C]
    positions = [rand(3) for _ in 1:4]
    magnetic_moments = DFTK.normalize_magnetic_moment.(rand(4))

    system = DFTK.construct_system(lattice, atoms, positions, magnetic_moments)
    @test atomic_symbol(system) == [:Si, :C, :H, :C]
    @test bounding_box(system)  == collect(eachcol(lattice)) * u"bohr"
    @test position(system)      == [lattice * p * u"bohr" for p in positions]

    @test system[1].potential == Si
    @test system[2].potential == C
    @test system[3].potential == H
    @test system[4].potential == C
    @test !hasproperty(system[1], :pseudopotential)
    @test system[2].pseudopotential == "hgh/pbe/c-q4.hgh"
    @test system[3].pseudopotential == "hgh/lda/h-q1.hgh"
    @test system[4].pseudopotential == "hgh/pbe/c-q4.hgh"

    parsed = DFTK.parse_system(system)
    @test parsed.lattice   == lattice
    @test parsed.atoms     == atoms
    @test parsed.positions ≈ positions atol=1e-14
    @test parsed.magnetic_moments == magnetic_moments

    let system = attach_psp(system; family="hgh", functional="lda")
        for i in 1:4
            @test system[i].potential       == atoms[i]
            @test system[i].magnetic_moment == magnetic_moments[i]
        end
    end

    for constructor in (Model, model_atomic, model_LDA, model_PBE, model_SCAN)
        model = constructor(system)
        @test model.spin_polarization == :collinear
        newsys = atomic_system(model, magnetic_moments)

        @test atomic_symbol(system)       == atomic_symbol(newsys)
        @test bounding_box(system)        == bounding_box(newsys)
        @test boundary_conditions(system) == boundary_conditions(newsys)
        @test maximum(maximum, position(system) - position(newsys)) < 1e-12u"bohr"
        for (p, newp) in zip(system, newsys)
            @test p.magnetic_moment == newp.magnetic_moment
            @test p.potential       == newp.potential
        end
    end
end

@testset "AbstractSystem -> DFTK" begin
    lattice     = [12u"bohr" * rand(3) for _ in 1:3]
    atoms       = [:C => rand(3), :Si => rand(3), :H => rand(3), :C => rand(3)]
    pos_units   = last.(atoms)
    pos_lattice = austrip.(reduce(hcat, lattice))
    system      = periodic_system(atoms, lattice; fractional=true)

    let model = Model(system)
        @test model.lattice == pos_lattice
        @test model.positions ≈ pos_units atol=1e-14
        @test model.spin_polarization == :none

        @test length(model.atoms) == 4
        @test model.atoms[1] == ElementCoulomb(:C)
        @test model.atoms[2] == ElementCoulomb(:Si)
        @test model.atoms[3] == ElementCoulomb(:H)
        @test model.atoms[4] == ElementCoulomb(:C)
    end

    let system = attach_psp(system; family="hgh", functional="pbe")
        @test length(system) == 4
        @test system[1].pseudopotential == "hgh/pbe/c-q4.hgh"
        @test system[2].pseudopotential == "hgh/pbe/si-q4.hgh"
        @test system[3].pseudopotential == "hgh/pbe/h-q1.hgh"
        @test system[4].pseudopotential == "hgh/pbe/c-q4.hgh"

        parsed = DFTK.parse_system(system)
        @test parsed.lattice == pos_lattice
        @test parsed.positions ≈ pos_units atol=1e-14
        @test isempty(parsed.magnetic_moments)

        @test length(parsed.atoms) == 4
        @test parsed.atoms[1].psp.identifier == "hgh/pbe/c-q4.hgh"
        @test parsed.atoms[2].psp.identifier == "hgh/pbe/si-q4.hgh"
        @test parsed.atoms[3].psp.identifier == "hgh/pbe/h-q1.hgh"
        @test parsed.atoms[4].psp.identifier == "hgh/pbe/c-q4.hgh"
    end

    let system = attach_psp(system; family="hgh", functional="lda")
        @test length(system) == 4
        @test system[1].pseudopotential == "hgh/lda/c-q4.hgh"
        @test system[2].pseudopotential == "hgh/lda/si-q4.hgh"
        @test system[3].pseudopotential == "hgh/lda/h-q1.hgh"
        @test system[4].pseudopotential == "hgh/lda/c-q4.hgh"

        model = Model(system)
        @test model.lattice == pos_lattice
        @test model.positions ≈ pos_units atol=1e-14
        @test model.spin_polarization == :none

        @test length(model.atoms) == 4
        @test model.atoms[1].psp.identifier == "hgh/lda/c-q4.hgh"
        @test model.atoms[2].psp.identifier == "hgh/lda/si-q4.hgh"
        @test model.atoms[3].psp.identifier == "hgh/lda/h-q1.hgh"
        @test model.atoms[4].psp.identifier == "hgh/lda/c-q4.hgh"
    end
end


@testset "Check attach_psp routine selectively" begin
    symbols = [:Cu, :Au, :Ni]

    let pspmap = compute_pspmap(symbols; functional="lda", family="hgh", core=:semicore)
        @test pspmap[:Cu] == "hgh/lda/cu-q19.hgh"
        @test pspmap[:Au] == "hgh/lda/au-q19.hgh"
        @test pspmap[:Ni] == "hgh/lda/ni-q18.hgh"
    end

    let pspmap = compute_pspmap(symbols; functional="lda", family="hgh", core=:fullcore)
        @test pspmap[:Cu] == "hgh/lda/cu-q11.hgh"
        @test pspmap[:Au] == "hgh/lda/au-q11.hgh"
        @test pspmap[:Ni] == "hgh/lda/ni-q10.hgh"
    end

    let pspmap = compute_pspmap([:Cu, :Au]; functional="pbe", family="hgh", core=:semicore)
        @test pspmap[:Cu] == "hgh/pbe/cu-q19.hgh"
        @test pspmap[:Au] == "hgh/pbe/au-q19.hgh"
    end

    let pspmap = compute_pspmap(symbols; functional="pbe", family="hgh", core=:fullcore)
        @test pspmap[:Cu] == "hgh/pbe/cu-q11.hgh"
        @test pspmap[:Au] == "hgh/pbe/au-q11.hgh"
        @test pspmap[:Ni] == "hgh/pbe/ni-q18.hgh"
    end
end
