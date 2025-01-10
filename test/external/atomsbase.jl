@testitem "parse_system and DFTK -> AbstractSystem -> DFTK" tags=[:atomsbase] begin
    using DFTK
    using Unitful
    using UnitfulAtomic
    using AtomsBase
    using PseudoPotentialData

    family = PseudoFamily("dojo.nc.sr.pbe.v0_4_1.standard.upf")
    Si = ElementCoulomb(:Si)
    C  = ElementPsp(:C, family)
    H  = ElementPsp(:H, family)

    lattice   = randn(3, 3)
    atoms     = [Si, C, H, C]
    positions = [rand(3) for _ = 1:4]
    magnetic_moments = rand(4)

    system = atomic_system(lattice, atoms, positions, magnetic_moments)
    @test atomic_symbol(system, :) == [:Si, :C, :H, :C]
    @test mass(system, :)       == [28.085u"u", 12.011u"u", 1.008u"u", 12.011u"u"]
    @test cell_vectors(system)  == Tuple(v * u"bohr" for v in eachcol(lattice))
    @test position(system, :)   == [lattice * p * u"bohr" for p in positions]
    @test system[:, :magnetic_moment] == magnetic_moments

    @testset "Parsing system without pseudopotentials" begin
        parsed = DFTK.parse_system(system, fill(nothing, length(atoms)))
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

    @testset "Parsing system with dictionary of explicit paths" begin
        gth = PseudoFamily("cp2k.nc.sr.lda.v0_1.largecore.gth")
        pspmap = Dict(:H => family[:H], :Si => family[:Si], :C => gth[:C])
        parsed = DFTK.parse_system(system, pspmap)
        @test length(parsed.atoms) == 4
        # Identifier is filename, but on windows we replace backslash path
        # delimiter by forward slash to homogenise the identifier
        @test parsed.atoms[1].psp.identifier == replace(pspmap[:Si], "\\" => "/")
        @test parsed.atoms[2].psp.identifier == replace(pspmap[:C],  "\\" => "/")
        @test parsed.atoms[3].psp.identifier == replace(pspmap[:H],  "\\" => "/")
        @test parsed.atoms[4].psp.identifier == replace(pspmap[:C],  "\\" => "/")
    end

    @testset "Constructing model with pseudo family" begin
        model = model_atomic(system; pseudopotentials=family)
        @test length(model.atoms) == 4
        @test model.atoms[1].psp.identifier == replace(family[:Si], "\\" => "/")
        @test model.atoms[2].psp.identifier == replace(family[:C],  "\\" => "/")
        @test model.atoms[3].psp.identifier == replace(family[:H],  "\\" => "/")
        @test model.atoms[4].psp.identifier == replace(family[:C],  "\\" => "/")
    end

    @testset "system -> Model -> system" begin
        for constructor in (Model, model_atomic)
            model = constructor(system; pseudopotentials=family)
            @test model.spin_polarization == :collinear
            newsys = periodic_system(model, magnetic_moments)

            @test atomic_symbol(system, :) == atomic_symbol(newsys, :)
            @test mass(system, :)          == mass(newsys, :)
            @test cell_vectors(system)     == cell_vectors(newsys)
            @test periodicity(system)      == periodicity(newsys)
            @test maximum(maximum, position(system, :) - position(newsys, :)) < 1e-12u"bohr"
            @test system[:, :magnetic_moment] == newsys[:, :magnetic_moment]
        end
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

    pbegth = PseudoFamily("dojo.nc.sr.pbe.v0_4_1.standard.upf")
    ldagth = PseudoFamily("cp2k.nc.sr.lda.v0_1.largecore.gth")
    pspmap = Dict(:H => pbegth[:H], :Si => pbegth[:Si], :C => ldagth[:C])
    let model = Model(system; pseudopotentials=pspmap)
        @test model.lattice   ≈ pos_lattice atol=1e-12
        @test model.positions ≈ pos_units   atol=1e-12
        @test model.spin_polarization == :none

        @test length(model.atoms) == 4
        @test model.atoms[1].psp.identifier == replace(pspmap[:C],  "\\" => "/")
        @test model.atoms[2].psp.identifier == replace(pspmap[:Si], "\\" => "/")
        @test model.atoms[3].psp.identifier == replace(pspmap[:H],  "\\" => "/")
        @test model.atoms[4].psp.identifier == replace(pspmap[:C],  "\\" => "/")
    end

    let
        gth = PseudoFamily("cp2k.nc.sr.lda.v0_1.largecore.gth")
        psp_Si = load_psp(gth, :Si)
        psp_H  = load_psp(gth, :H)
        psp_C  = load_psp(gth, :C)
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

    let family = PseudoFamily("dojo.nc.sr.pbe.v0_4_1.standard.upf")
        model = Model(system; pseudopotentials=family)

        @test model.lattice   ≈ pos_lattice atol=1e-12
        @test model.positions ≈ pos_units   atol=1e-12
        @test model.spin_polarization == :none

        # Identifier is filename, but on windows we replace backslash path
        # delimiter by forward slash to homogenise the identifier
        @test length(model.atoms) == 4
        @test model.atoms[1].psp.identifier == replace(family[:C],  "\\" => "/")
        @test model.atoms[2].psp.identifier == replace(family[:Si], "\\" => "/")
        @test model.atoms[3].psp.identifier == replace(family[:H],  "\\" => "/")
        @test model.atoms[4].psp.identifier == replace(family[:C],  "\\" => "/")
    end
end

@testitem "AbstractSystem (unusual symbols and masses) -> DFTK" tags=[:atomsbase] begin
    using DFTK
    using Unitful
    using UnitfulAtomic
    using AtomsBase
    using PseudoPotentialData

    lattice = [12u"bohr" * rand(3) for _ = 1:3]
    atoms   = [Atom(6, randn(3)u"Å"; species=ChemicalSpecies(:C12), mass=-1u"u"),
               Atom(6, randn(3)u"Å"; species=ChemicalSpecies(:C),   mass=-2u"u")]
    system  = periodic_system(atoms, lattice)

    gth = PseudoFamily("cp2k.nc.sr.lda.v0_1.largecore.gth")
    pseudopotentials = Dict(:C => gth[:C])
    let model = model_DFT(system; functionals=LDA(), pseudopotentials)
        @test model.lattice == austrip.(stack(lattice))
        @test model.lattice * model.positions[1] * u"bohr" ≈ atoms[1].position
        @test model.lattice * model.positions[2] * u"bohr" ≈ atoms[2].position
        @test model.spin_polarization == :none

        @test length(model.atoms) == 2
        @test element_symbol(model.atoms[1]) == :C
        @test element_symbol(model.atoms[2]) == :C
        @test mass.(model.atoms) == [-1u"u", -2u"u"]
        @test model.atoms[1].psp.identifier == replace(gth[:C], "\\" -> "/")
        @test model.atoms[2].psp.identifier == replace(gth[:C], "\\" -> "/")
    end
end
