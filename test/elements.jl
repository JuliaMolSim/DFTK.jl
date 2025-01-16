@testitem "Check constructing ElementCoulomb" begin
    using DFTK
    using Unitful
    using UnitfulAtomic
    using AtomsBase
    using DFTK: charge_nuclear, charge_ionic, n_elec_core, n_elec_valence
    using DFTK: ElementCoulomb, local_potential_fourier, local_potential_real
    using LinearAlgebra

    el_by_symbol = ElementCoulomb(:O)
    @test el_by_symbol.species == ChemicalSpecies(:O)
    @test charge_nuclear(el_by_symbol) == 8
    @test element_symbol(el_by_symbol) == :O
    el_by_number = ElementCoulomb(14)
    @test element_symbol(el_by_number) == :Si

    element = ElementCoulomb(:Mg)
    @test element.species == ChemicalSpecies(:Mg)
    @test species(element) == ChemicalSpecies(:Mg)
    @test element_symbol(element) == :Mg
    @test mass(element) == 24.305u"u"
    @test charge_nuclear(element) == 12
    @test charge_ionic(element) == 12
    @test n_elec_valence(element) == charge_ionic(element)
    @test n_elec_core(element) == 0

    @test local_potential_fourier(element, 0.0) == 0.0
    @test local_potential_fourier(element, norm([2.0, 0, 0])) == -12π
    @test local_potential_real(element, norm([2.0, 0, 0])) == -6.0
end

@testitem "Check constructing ElementPsp" begin
    using DFTK
    using Unitful
    using UnitfulAtomic
    using AtomsBase
    using PseudoPotentialData
    using DFTK: load_psp, charge_nuclear, charge_ionic, n_elec_core, n_elec_valence
    using DFTK: ElementPsp, local_potential_fourier, local_potential_real

    pseudopotentials = PseudoFamily("cp2k.nc.sr.lda.v0_1.largecore.gth")
    el_by_symbol = ElementPsp(:W, pseudopotentials)
    @test charge_nuclear(el_by_symbol) == 74
    @test element_symbol(el_by_symbol) == :W
    el_by_number = ElementPsp(1, pseudopotentials)
    @test el_by_number.species == ChemicalSpecies(:H)

    element = ElementPsp(:C, pseudopotentials)
    @test species(element) == ChemicalSpecies(:C)
    @test element.psp !== nothing
    @test element.psp.identifier == replace(pseudopotentials[:C], "\\" => "/")

    @test mass(element) == 12.011u"u"
    @test element_symbol(element) == :C
    @test charge_nuclear(element) == 6
    @test charge_ionic(element) == 4
    @test n_elec_valence(element) == 4
    @test n_elec_core(element) == 2

    @test local_potential_fourier(element, 0.0) == 0.0
    @test local_potential_fourier(element, 2.0) == -12.695860686869914
    @test local_potential_real(element, 2.0) == -1.999997661838144
end

@testitem "Check constructing ElementPsp from family" begin
    using DFTK
    using PseudoPotentialData
    pd_lda_family = PseudoFamily("dojo.nc.sr.lda.v0_4_1.standard.upf")

    element_from_family = ElementPsp(:Si, pd_lda_family)
    element_explicit_rcut = ElementPsp(:Si, pd_lda_family; rcut=15)
    element_from_psp = ElementPsp(:Si, load_psp(pd_lda_family[:Si]))

    # Constructing a PSP from a PD family should give the right rcut (10)
    @test element_from_family.psp.rcut == 10
    # Overriding the rcut takes precedence
    @test element_explicit_rcut.psp.rcut == 15
    # Constructing a PSP from a file cannot infer the rcut
    @test element_from_family.psp.rcut != element_from_psp.psp.rcut
end

@testitem "Check constructing ElementCohenBergstresser" begin
    using DFTK
    using Unitful
    using UnitfulAtomic
    using AtomsBase
    using DFTK: charge_nuclear, charge_ionic, n_elec_core, n_elec_valence
    using DFTK: ElementCohenBergstresser, local_potential_fourier

    element_Ge = ElementCohenBergstresser(:Ge)
    @test species(element_Ge) == ChemicalSpecies(:Ge)

    element = ElementCohenBergstresser(:Si)
    @test species(element) == ChemicalSpecies(:Si)

    @test element_symbol(element) == :Si
    @test mass(element) == 28.085u"u"
    @test charge_nuclear(element) == 14
    @test charge_ionic(element) == 4
    @test n_elec_valence(element) == 4
    @test n_elec_core(element) == 10

    @test local_potential_fourier(element, 0.0) == 0.0
    p3 = sqrt(3) * 2π / element.lattice_constant
    @test local_potential_fourier(element, p3) == -14.180625963358901
end

@testitem "Check constructing ElementGaussian" begin
    using DFTK
    using Unitful
    using UnitfulAtomic
    using AtomsBase
    using DFTK: local_potential_fourier, local_potential_real

    element = ElementGaussian(1.0, 0.5; symbol=:X1)

    @test element_symbol(element) == :X1
    @test isnothing(mass(element))
    @test charge_nuclear(element) == 0
    @test isnothing(species(element))
    @test local_potential_fourier(element, 0.0) == -1.0
    @test local_potential_fourier(element, 2.0) == -0.6065306597126334
    @test local_potential_real(element, 2.0) == -0.00026766045152977074
end
