@testitem "Check constructing ElementCoulomb" begin
    using DFTK
    using Unitful
    using UnitfulAtomic
    using DFTK: charge_nuclear, charge_ionic, n_elec_core, n_elec_valence
    using DFTK: ElementCoulomb, local_potential_fourier, local_potential_real
    using LinearAlgebra

    el_by_name = ElementCoulomb("oxygen")
    @test el_by_name.Z == 8
    @test el_by_name.symbol == :O
    el_by_number = ElementCoulomb(14)
    @test el_by_number.symbol == :Si

    element = ElementCoulomb(:Mg)
    @test element.Z == 12
    @test element.symbol == :Mg

    @test atomic_symbol(element) == :Mg
    @test atomic_mass(element) == 24.305u"u"
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
    using DFTK: load_psp, charge_nuclear, charge_ionic, n_elec_core, n_elec_valence
    using DFTK: ElementPsp, local_potential_fourier, local_potential_real

    el_by_name = ElementPsp("tungsten"; psp=load_psp("hgh/lda/w-q6"))
    @test el_by_name.Z == 74
    @test el_by_name.symbol == :W
    el_by_number = ElementPsp(1; psp=load_psp("hgh/pbe/H-q1"))
    @test el_by_number.symbol == :H

    element = ElementPsp("carbon"; psp=load_psp("hgh/lda/C-q4"))

    @test element.Z == 6
    @test element.symbol == :C
    @test element.psp !== nothing
    @test element.psp.identifier == "hgh/lda/c-q4"

    @test atomic_symbol(element) == :C
    @test atomic_mass(element) == 12.011u"u"
    @test charge_nuclear(element) == 6
    @test charge_ionic(element) == 4
    @test n_elec_valence(element) == 4
    @test n_elec_core(element) == 2

    @test local_potential_fourier(element, 0.0) == 0.0
    @test local_potential_fourier(element, 2.0) == -12.695860686869914
    @test local_potential_real(element, 2.0) == -1.999997661838144
end

@testitem "Check constructing ElementCohenBergstresser" begin
    using DFTK
    using Unitful
    using UnitfulAtomic
    using DFTK: charge_nuclear, charge_ionic, n_elec_core, n_elec_valence
    using DFTK: ElementCohenBergstresser, local_potential_fourier

    element_Ge = ElementCohenBergstresser(:Ge)
    @test element_Ge.Z == 32

    element = ElementCohenBergstresser("silicon")
    @test element.Z == 14
    @test element.symbol == :Si

    @test atomic_symbol(element) == :Si
    @test atomic_mass(element) == 28.085u"u"
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
    using DFTK: local_potential_fourier, local_potential_real

    element = ElementGaussian(1.0, 0.5; symbol=:X1)

    @test atomic_symbol(element) == :X1
    @test isnothing(atomic_mass(element))
    @test local_potential_fourier(element, 0.0) == -1.0
    @test local_potential_fourier(element, 2.0) == -0.6065306597126334
    @test local_potential_real(element, 2.0) == -0.00026766045152977074
end
