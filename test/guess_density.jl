@testitem "Guess density integrates to number of electrons" setup=[TestCases] begin
    using DFTK
    silicon = TestCases.silicon

    function build_basis(atoms, spin_polarization)
        model = model_LDA(silicon.lattice, atoms, silicon.positions; spin_polarization,
                          temperature=0.01)
        PlaneWaveBasis(model; Ecut=7, kgrid=[3, 3, 3], kshift=[1, 1, 1] / 2)
    end
    total_charge(basis, ρ) = sum(ρ) * basis.model.unit_cell_volume / prod(basis.fft_size)

    Si_upf = ElementPsp(silicon.atnum; psp=load_psp(silicon.psp_upf))
    Si_hgh = ElementPsp(silicon.atnum; psp=load_psp(silicon.psp_hgh))
    magnetic_moments = [1.0, -1.0]
    methods  = [ValenceDensityGaussian(), ValenceDensityPseudo(), ValenceDensityAuto()]
    elements = [[Si_upf, Si_hgh], [Si_upf, Si_upf], [Si_upf, Si_hgh]]

    @testset "Random" begin
        method = RandomDensity()
        basis = build_basis([Si_upf, Si_hgh], :none)
        ρ = @inferred guess_density(basis, method)
        @test total_charge(basis, ρ) ≈ basis.model.n_electrons

        basis = build_basis([Si_upf, Si_hgh], :collinear)
        ρ = @inferred guess_density(basis, method)
        @test total_charge(basis, ρ) ≈ basis.model.n_electrons
    end

    @testset "Atomic $(string(typeof(method)))" for (method, elements) in zip(methods, elements)
        basis = build_basis(elements, :none)
        ρ = @inferred guess_density(basis, method)
        @test total_charge(basis, ρ) ≈ basis.model.n_electrons

        basis = build_basis(elements, :collinear)
        ρ = @inferred guess_density(basis, method)
        @test total_charge(basis, ρ) ≈ basis.model.n_electrons

        ρ = @inferred guess_density(basis, method, magnetic_moments)
        @test total_charge(basis, ρ) ≈ basis.model.n_electrons
    end
end
