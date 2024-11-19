@testitem "parse_system and DFTK -> AbstractSystem -> DFTK" tags=[:atomsbase] begin
    using DFTK
    using Unitful
    using UnitfulAtomic
    using AtomsBase
    using PseudoPotentialData

    Si = ElementCoulomb(:Si)
    C  = ElementPsp(:C, load_psp("hgh/pbe/c-q4.hgh"))
    H  = ElementPsp(:H, load_psp("hgh/lda/h-q1.hgh"))

    lattice   = randn(3, 3)
    atoms     = [Si, C, H, C]
    positions = [rand(3) for _ = 1:4]
    magnetic_moments = rand(4)

    system = atomic_system(lattice, atoms, positions, magnetic_moments)
    @test atomic_symbol(system) == [:Si, :C, :H, :C]
    @test atomic_mass(system)   == [28.085u"u", 12.011u"u", 1.008u"u", 12.011u"u"]
    @test bounding_box(system)  == collect(eachcol(lattice)) * u"bohr"
    @test position(system)      == [lattice * p * u"bohr" for p in positions]
    @test system[:, :magnetic_moment] == magnetic_moments

    let parsed = DFTK.parse_system(system, fill(nothing, length(atoms)))
        @test parsed.lattice   ≈ lattice   atol=1e-12
        @test parsed.positions ≈ positions atol=1e-12
        for i = 1:4
            @test iszero(parsed.magnetic_moments[i][1:2])
            @test parsed.magnetic_moments[i][3] == magnetic_moments[i]
        end
        @test length(parsed.atoms) == 4
        @test parsed.atoms[1] == ElementCoulomb(:Si)
        @test parsed.atoms[2] == ElementCoulomb(:C)
        @test parsed.atoms[3] == ElementCoulomb(:H)
        @test parsed.atoms[4] == ElementCoulomb(:C)
    end

    pspmap = Dict(:H => "hgh/pbe/h-q1.hgh", :Si => "hgh/pbe/si-q4.hgh",
                  :C => "hgh/pbe/c-q4.hgh")
    let parsed = DFTK.parse_system(system, pspmap)
        @test length(parsed.atoms) == 4
        @test parsed.atoms[1].psp.identifier == pspmap[:Si]
        @test parsed.atoms[2].psp.identifier == pspmap[:C]
        @test parsed.atoms[3].psp.identifier == pspmap[:H]
        @test parsed.atoms[4].psp.identifier == pspmap[:C]
    end

    family = PseudoFamily("pd_nc_sr_pbe_standard_0.4.1_upf")
    let model = model_atomic(system; pseudopotentials=family)
        @test length(model.atoms) == 4
        @test model.atoms[1].psp.identifier == family[:Si]
        @test model.atoms[2].psp.identifier == family[:C]
        @test model.atoms[3].psp.identifier == family[:H]
        @test model.atoms[4].psp.identifier == family[:C]
    end

    for constructor in (Model, model_atomic)
        model = constructor(system; pseudopotentials=family)
        @test model.spin_polarization == :collinear
        newsys = periodic_system(model, magnetic_moments)

        @test atomic_symbol(system)       == atomic_symbol(newsys)
        @test atomic_mass(system)         == atomic_mass(newsys)
        @test bounding_box(system)        == bounding_box(newsys)
        @test boundary_conditions(system) == boundary_conditions(newsys)
        @test maximum(maximum, position(system) - position(newsys)) < 1e-12u"bohr"
        @test system[:, :magnetic_moment] == newsys[:, :magnetic_moment]
    end
end

@testitem "DFTK -> AbstractSystem (noncollinear)" tags=[:atomsbase] begin
    using DFTK
    using AtomsBase

    lattice   = randn(3, 3)
    atoms     = [ElementCoulomb(:Si), ElementCoulomb(:C)]
    positions = [rand(3) for _ = 1:2]
    magnetic_moments = [rand(3), rand(3)]
    system = atomic_system(lattice, atoms, positions, magnetic_moments)
    @test system[:, :magnetic_moment] == magnetic_moments
end

@testitem "charged AbstractSystem -> DFTK" tags=[:atomsbase] begin
    using DFTK
    using Unitful
    using UnitfulAtomic
    using AtomsBase

    @testset "Charged system" begin
        lattice = [12u"bohr" * rand(3) for _ = 1:3]
        atoms   = [:C => rand(3), :Si => rand(3), :H => rand(3), :C => rand(3)]
        system  = periodic_system(atoms, lattice; fractional=true, charge=1.0u"e_au")
        @test_throws ErrorException Model(system)
    end

    @testset "Charged atoms, but neutral" begin
        lattice = [12u"bohr" * rand(3) for _ = 1:3]
        atoms   = [Atom(:C,  rand(3) * 12u"bohr", charge=1.0u"e_au"),
                   Atom(:Si, rand(3) * 12u"bohr", charge=-1.0u"e_au")]
        system  = periodic_system(atoms, lattice)
        model   = Model(system)
        @test model.n_electrons == 6 + 14
    end

    @testset "Charged atoms and not neutral" begin
        lattice = [12u"bohr" * rand(3) for _ = 1:3]
        atoms   = [Atom(:C,  rand(3) * 12u"bohr", charge=1.0u"e_au"),
                   Atom(:Si, rand(3) * 12u"bohr", charge=-2.0u"e_au")]
        system  = periodic_system(atoms, lattice)
        @test_throws ErrorException Model(system)
    end
end

@testitem "AbstractSystem -> DFTK Model" tags=[:atomsbase] begin
    using DFTK
    using Unitful
    using UnitfulAtomic
    using AtomsBase
    using PseudoPotentialData

    lattice     = [12u"bohr" * rand(3) for _ = 1:3]
    weirdatom   = Atom(6, randn(3)u"Å"; atomic_symsymbol=:C1)
    atoms       = [:C => rand(3), :Si => rand(3), :H => rand(3), :C => rand(3)]
    pos_units   = last.(atoms)
    pos_lattice = austrip.(stack(lattice))
    system      = periodic_system(atoms, lattice; fractional=true)

    let model = Model(system)
        @test model.lattice   ≈ pos_lattice atol=1e-12
        @test model.positions ≈ pos_units   atol=1e-12
        @test model.spin_polarization == :none

        @test length(model.atoms) == 4
        @test model.atoms[1] == ElementCoulomb(:C)
        @test model.atoms[2] == ElementCoulomb(:Si)
        @test model.atoms[3] == ElementCoulomb(:H)
        @test model.atoms[4] == ElementCoulomb(:C)
    end

    pbemap = Dict(:H => "hgh/pbe/h-q1.hgh", :Si => "hgh/pbe/si-q4.hgh",
                  :C => "hgh/pbe/c-q4.hgh")
    let model = Model(system; pseudopotentials=pbemap)
        @test model.lattice   ≈ pos_lattice atol=1e-12
        @test model.positions ≈ pos_units   atol=1e-12
        @test model.spin_polarization == :none

        @test length(model.atoms) == 4
        @test model.atoms[1].psp.identifier == "hgh/pbe/c-q4.hgh"
        @test model.atoms[2].psp.identifier == "hgh/pbe/si-q4.hgh"
        @test model.atoms[3].psp.identifier == "hgh/pbe/h-q1.hgh"
        @test model.atoms[4].psp.identifier == "hgh/pbe/c-q4.hgh"
    end

    let
        psp_Si = load_psp("hgh/pbe/si-q4.hgh")
        psp_H  = load_psp("hgh/lda/h-q1.hgh")
        psp_C  = load_psp("hgh/pbe/c-q4.hgh")
        model = Model(system; pseudopotentials=[nothing, psp_Si, psp_H, psp_C])

        @test model.lattice   ≈ pos_lattice atol=1e-12
        @test model.positions ≈ pos_units   atol=1e-12
        @test model.spin_polarization == :none

        @test length(model.atoms) == 4
        @test model.atoms[1] == ElementCoulomb(:C)
        @test model.atoms[2] == ElementPsp(:Si, psp_Si)
        @test model.atoms[3] == ElementPsp(:H,  psp_H)
        @test model.atoms[4] == ElementPsp(:C,  psp_C)
    end

    let family = PseudoFamily("pd_nc_sr_pbe_standard_0.4.1_upf")
        model = Model(system; pseudopotentials=family)

        @test model.lattice   ≈ pos_lattice atol=1e-12
        @test model.positions ≈ pos_units   atol=1e-12
        @test model.spin_polarization == :none

        @test length(model.atoms) == 4
        @test model.atoms[1].psp.identifier == family[:C]
        @test model.atoms[2].psp.identifier == family[:Si]
        @test model.atoms[3].psp.identifier == family[:H]
        @test model.atoms[4].psp.identifier == family[:C]
    end
end

@testitem "AbstractSystem (unusual symbols and masses) -> DFTK" tags=[:atomsbase] begin
    using DFTK
    using Unitful
    using UnitfulAtomic
    using AtomsBase

    lattice = [12u"bohr" * rand(3) for _ = 1:3]
    # Later with AtomsBase 0.5
    # atoms   = [Atom(6, randn(3)u"Å"; species=ChemicalSpecies(:C12), mass=-1u"u"),
    #            Atom(6, randn(3)u"Å"; species=ChemicalSpecies(:C),   mass=-2u"u")]
    atoms   = [Atom(6, randn(3)u"Å"; atomic_symbol=:C, atomic_mass=-1u"u"),
               Atom(6, randn(3)u"Å"; atomic_symbol=:C, atomic_mass=-2u"u")]
    system  = periodic_system(atoms, lattice)

    pseudopotentials = Dict(:C => "hgh/lda/c-q4.hgh")
    let model = model_DFT(system; functionals=LDA(), pseudopotentials)
        @test model.lattice == austrip.(stack(lattice))
        @test model.lattice * model.positions[1] * u"bohr" ≈ atoms[1].position
        @test model.lattice * model.positions[2] * u"bohr" ≈ atoms[2].position
        @test model.spin_polarization == :none

        @test length(model.atoms) == 2
        @test atomic_symbol(model.atoms[1]) == :C
        @test atomic_symbol(model.atoms[2]) == :C
        @test atomic_mass.(model.atoms) == [-1u"u", -2u"u"]
        @test model.atoms[1].psp.identifier == "hgh/lda/c-q4.hgh"
        @test model.atoms[2].psp.identifier == "hgh/lda/c-q4.hgh"
    end
end
