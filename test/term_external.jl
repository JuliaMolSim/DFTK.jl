using Test
using DFTK
using LinearAlgebra: dot

include("testcases.jl")

@warn "term_external.jl disabled for now"
if false
@testset "term_external using Coulomb potential" begin
    Ecut = 4
    fft_size = [15, 15, 15]
    model = Model(silicon.lattice, n_electrons=silicon.n_electrons)
    basis = PlaneWaveBasis(model, Ecut, silicon.kcoords, silicon.ksymops; fft_size=fft_size)
    function build_external(atoms)
        _, pot = term_external(atoms)(basis, nothing, zeros(ComplexF64, basis.fft_size))
        pot
    end

    @testset "Construction using a single function" begin
        function pot_coulomb(G)
            norm(G) == 0 && return 0.0
            -12*4π / sum(abs2, model.recip_lattice * G)
        end

        # Shifting by a lattice vector should not make a difference:
        pot0 = build_external([pot_coulomb => [[0, 0, 0]]])
        pot1 = build_external([pot_coulomb => [[0, 1, 0]]])
        @test pot0 ≈ pot1

        # Results are additive
        pot1 = build_external([pot_coulomb => [[0, 1/3, 0]]])
        pot2 = build_external([pot_coulomb => [[0, 1/3, 0], [0,0,0]]])
        @test pot0 + pot1 ≈ pot2

        # pot3 back to the PW basis to check we get the 1/|G|^2 behaviour
        pot3 = build_external([pot_coulomb => [[0, 1/8, 0]]])
        values_fourier = r_to_G(basis, complex(pot3))

        lattice_vector = silicon.lattice * [0, 1/8, 0]
        reference = [(1/sqrt(model.unit_cell_volume) * pot_coulomb(G)
                      * cis(dot(-model.recip_lattice * G, lattice_vector)))
                     for G in G_vectors(basis)]
        reference[1] = 0
        @test reference ≈ values_fourier
    end

    @testset "Construction using multiple functions" begin
        function pot_coulomb14(G)
            norm(G) == 0 && return 0.0
            return -14*4π / sum(abs2, model.recip_lattice * G)
        end
        function pot_coulomb6(G)
            norm(G) == 0 && return 0.0
            return -6*4π / sum(abs2, model.recip_lattice * G)
        end

        # Compute separate potential terms for reference
        pot_Si_1 = build_external([pot_coulomb14 => [[0, 1/8, 0]]])
        pot_Si_2 = build_external([pot_coulomb14 => [[0, -1/8, 0]]])
        pot_C = build_external([pot_coulomb6 => [[0, 1/4, 0]]])
        reference = pot_Si_1 + pot_Si_2 + pot_C

        pot = build_external([pot_coulomb14 => [[0, 1/8, 0], [0, -1/8, 0]],
                              pot_coulomb6 => [[0, 1/4, 0]]])
        @test pot ≈ reference
    end
end


@testset "term_external using elements" begin
    Ecut = 4
    fft_size = [15, 15, 15]
    model = Model(silicon.lattice, n_electrons=silicon.n_electrons)
    basis = PlaneWaveBasis(model, Ecut, silicon.kcoords, silicon.ksymops; fft_size=fft_size)
    function build_external(atoms)
        _, pot = term_external(atoms)(basis, nothing, zeros(ComplexF64, basis.fft_size))
        pot
    end

    @testset "Test without Pseudopotential" begin
        function pot_coulomb(G)
            norm(G) == 0 && return 0.0
            -14*4π / sum(abs2, model.recip_lattice * G)
        end
        silicon = ElementCoulomb(:Si)

        ref = build_external([pot_coulomb => [[0, 0, 0], [0, 1/3, 0]]])
        pot = build_external([silicon => [[0, 0, 0], [0, 1/3, 0]]])
        @test pot ≈ ref
    end
end
end
