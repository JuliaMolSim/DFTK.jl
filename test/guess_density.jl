using Test
using DFTK
include("testcases.jl")

@testset "Guess density integrates to number of electrons" begin
    function build_basis(atoms, spin_polarization)
        model = model_LDA(silicon.lattice, atoms, silicon.positions; spin_polarization,
                          temperature=0.01)
        PlaneWaveBasis(model; Ecut=7, kgrid=[3, 3, 3], kshift=[1, 1, 1] / 2)
    end
    total_charge(basis, ρ) = sum(ρ) * basis.model.unit_cell_volume / prod(basis.fft_size)


    Si_upf = ElementPsp(silicon.atnum, psp=load_psp(silicon.psp_upf))
    Si_hgh = ElementPsp(silicon.atnum, psp=load_psp(silicon.psp_hgh))

    @testset "Random" begin
        basis = build_basis([Si_upf, Si_hgh], :none)
        ρ = guess_density(basis; method=RandomGuessDensity())
        @test total_charge(basis, ρ) ≈ basis.model.n_electrons
    
        basis = build_basis([Si_upf, Si_hgh], :collinear)
        ρ = guess_density(basis; method=RandomGuessDensity())
        @test total_charge(basis, ρ) ≈ basis.model.n_electrons
    end

    @testset "Gaussian" begin
        basis = build_basis([Si_upf, Si_hgh], :none)
        ρ = guess_density(basis; method=GaussianGuessDensity())
        @test total_charge(basis, ρ) ≈ basis.model.n_electrons
    
        basis = build_basis([Si_upf, Si_hgh], :collinear)
        ρ = guess_density(basis; method=GaussianGuessDensity())
        @test total_charge(basis, ρ) ≈ basis.model.n_electrons
    
        basis = basis
        ρ = guess_density(basis; method=GaussianGuessDensity(), magnetic_moments=[1.0, -1.0])
        @test total_charge(basis, ρ) ≈ basis.model.n_electrons
    end

    @testset "Pseudopotential" begin
        basis = build_basis([Si_upf, Si_upf], :none)
        ρ = guess_density(basis; method=PspGuessDensity())
        @test total_charge(basis, ρ) ≈ basis.model.n_electrons
    
        basis = build_basis([Si_upf, Si_upf], :collinear)
        ρ = guess_density(basis; method=PspGuessDensity())
        @test total_charge(basis, ρ) ≈ basis.model.n_electrons
    
        basis = build_basis([Si_upf, Si_upf], :collinear)
        ρ = guess_density(basis; method=PspGuessDensity(), magnetic_moments=[1.0, -1.0])
        @test total_charge(basis, ρ) ≈ basis.model.n_electrons
    end

    @testset "Auto" begin
        basis = build_basis([Si_upf, Si_hgh], :none)
        ρ = guess_density(basis; method=AutoGuessDensity())
        @test total_charge(basis, ρ) ≈ basis.model.n_electrons
    
        basis = build_basis([Si_upf, Si_hgh], :collinear)
        ρ = guess_density(basis; method=AutoGuessDensity())
        @test total_charge(basis, ρ) ≈ basis.model.n_electrons
    
        basis = build_basis([Si_upf, Si_hgh], :collinear)
        ρ = guess_density(basis; method=AutoGuessDensity(), magnetic_moments=[1.0, -1.0])
        @test total_charge(basis, ρ) ≈ basis.model.n_electrons
    end
end
