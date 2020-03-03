using Test
using DFTK: load_psp, charge_nuclear, charge_ionic, n_elec_core, n_elec_valence
using DFTK: ElementPsp, ElementCohenBergstresser, ElementAllElectron
using DFTK: local_potential_fourier, local_potential_real

@testset "Check constructing ElementAllElectron" begin
    el_by_name = ElementAllElectron("oxygen")
    @test el_by_name.Z == 8
    @test el_by_name.symbol == :O
    el_by_number = ElementAllElectron(14)
    @test el_by_number.symbol == :Si

    element = ElementAllElectron(:Mg)
    @test element.Z == 12
    @test element.symbol == :Mg

    @test charge_nuclear(element) == 12
    @test charge_ionic(element) == 12
    @test n_elec_valence(element) == charge_ionic(element)
    @test n_elec_core(element) == 0

    @test local_potential_fourier(element, 0.0) == 0.0
    @test local_potential_fourier(element, [2.0, 0, 0]) == -12π
    @test local_potential_real(element, [2.0, 0, 0]) == -6.0
end

@testset "Check constructing ElementPsp" begin
    el_by_name = ElementPsp("tungsten", psp=load_psp("hgh/lda/w-q6"))
    @test el_by_name.Z == 74
    @test el_by_name.symbol == :W
    el_by_number = ElementPsp(1, psp=load_psp("hgh/pbe/H-q1"))
    @test el_by_number.symbol == :H

    element = ElementPsp("carbon", psp=load_psp("hgh/lda/C-q4"))

    @test element.Z == 6
    @test element.symbol == :C
    @test element.psp !== nothing
    @test element.psp.identifier == "hgh/lda/c-q4"

    @test charge_nuclear(element) == 6
    @test charge_ionic(element) == 4
    @test n_elec_valence(element) == 4
    @test n_elec_core(element) == 2

    @test local_potential_fourier(element, 0.0) == 0.0
    @test local_potential_fourier(element, 2.0) == -12.695860686869912
    @test local_potential_real(element, 2.0) == -1.981967787205101
end

@testset "Check constructing ElementCohenBergstresser" begin
    element = ElementCohenBergstresser("silicon")
    @test element.Z == 14
    @test element.symbol == :Si

    @test charge_nuclear(element) == 14
    @test charge_ionic(element) == 2
    @test n_elec_valence(element) == 2
    @test n_elec_core(element) == 12

    @test local_potential_fourier(element, 0.0) == 0.0
    q3 = sqrt(3) * 2π / element.lattice_constant
    @test local_potential_fourier(element, q3) == -14.18062598209035
end
