using Test
using LinearAlgebra: norm
using DFTK: n_elec_core, n_elec_valence
using DFTK: ElementPsp, ElementCohenBergstresser, ElementCoulomb
import PseudoPotentialIO

@testset "Check constructing ElementCoulomb" begin
    el_by_name = ElementCoulomb("oxygen")
    @test el_by_name.Z == 8
    @test el_by_name.symbol == :O
    el_by_number = ElementCoulomb(14)
    @test el_by_number.symbol == :Si

    element = ElementCoulomb(:Mg)
    @test element.Z == 12
    @test element.symbol == :Mg

    @test atomic_symbol(element) == :Mg
    @test PseudoPotentialIO.atomic_charge(element) == 12
    @test PseudoPotentialIO.PseudoPotentialIO.valence_charge(element) == 12
    @test n_elec_valence(element) == PseudoPotentialIO.PseudoPotentialIO.valence_charge(element)
    @test n_elec_core(element) == 0

    @test PseudoPotentialIO.local_potential_fourier(element)(0.0) == 0.0
    @test PseudoPotentialIO.local_potential_fourier(element)(norm([2.0, 0, 0])) == -12π
    @test PseudoPotentialIO.local_potential_real(element)(norm([2.0, 0, 0])) == -6.0
end

@testset "Check constructing ElementPsp" begin
    el_by_name = ElementPsp("tungsten", psp=PseudoPotentialIO.load_psp("hgh_lda_hgh", "w-q6.hgh"))
    @test el_by_name.Z == 74
    @test el_by_name.symbol == :W
    el_by_number = ElementPsp(1, psp=PseudoPotentialIO.load_psp("hgh_pbe_hgh", "h-q1.hgh"))
    @test el_by_number.symbol == :H

    element = ElementPsp("carbon", psp=PseudoPotentialIO.load_psp("hgh_lda_hgh", "c-q4.hgh"))

    @test element.Z == 6
    @test element.symbol == :C
    @test element.psp !== nothing

    @test atomic_symbol(element) == :C
    @test PseudoPotentialIO.atomic_charge(element) == 6
    @test PseudoPotentialIO.PseudoPotentialIO.valence_charge(element) == 4
    @test n_elec_valence(element) == 4
    @test n_elec_core(element) == 2

    @test PseudoPotentialIO.local_potential_fourier(element)(0.0) == 0.0
    @test PseudoPotentialIO.local_potential_fourier(element)(2.0) == -12.695860686869914
    @test PseudoPotentialIO.local_potential_real(element)(2.0) == -1.999997661838144
end

@testset "Check constructing ElementCohenBergstresser" begin
    element_Ge = ElementCohenBergstresser(:Ge)
    @test element_Ge.Z == 32

    element = ElementCohenBergstresser("silicon")
    @test element.Z == 14
    @test element.symbol == :Si

    @test atomic_symbol(element) == :Si
    @test PseudoPotentialIO.atomic_charge(element) == 14
    @test PseudoPotentialIO.PseudoPotentialIO.valence_charge(element) == 4
    @test n_elec_valence(element) == 4
    @test n_elec_core(element) == 10

    @test PseudoPotentialIO.local_potential_fourier(element)(0.0) == 0.0
    q3 = sqrt(3) * 2π / element.lattice_constant
    @test PseudoPotentialIO.local_potential_fourier(element)(q3) == -14.180625963358901
end

@testset "Check constructing ElementGaussian" begin
    element = ElementGaussian(1.0, 0.5; symbol=:X1)

    @test atomic_symbol(element) == :X1

    @test PseudoPotentialIO.local_potential_fourier(element)(0.0) == -1.0
    @test PseudoPotentialIO.local_potential_fourier(element)(2.0) == -0.6065306597126334
    @test PseudoPotentialIO.local_potential_real(element)(2.0) == -0.00026766045152977074
end
