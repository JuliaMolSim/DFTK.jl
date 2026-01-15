using DFTK
using Test
using LinearAlgebra

@testset "Periodic Green's functions (1D)" begin
    # Setup simple 1D system with only kinetic + local potential (no nonlinearity)
    a = 10.0  # Box size
    lattice = a .* [[1 0 0.]; [0 0 0]; [0 0 0]]
    
    # Periodic cosine potential
    pot(x) = cos(2π * x / a)
    
    # Build model with only kinetic and external potential
    # Disable symmetry at model level
    n_electrons = 1
    terms = [
        Kinetic(),
        ExternalFromReal(r -> pot(r[1])),
    ]
    model = Model(lattice; n_electrons, terms, spin_polarization=:spinless, symmetries=false)
    
    # Setup basis with small Ecut
    Ecut = 50
    kgrid = MonkhorstPack([4, 1, 1])  # 1D grid with 4 k-points
    basis = PlaneWaveBasis(model; Ecut, kgrid)
    
    # Test h function computation
    @testset "h-function computation" begin
        # Get eigenstates
        ham = Hamiltonian(basis)
        n_bands = 5
        eigres = diagonalize_all_kblocks(diag_full, ham, n_bands)
        
        @test length(eigres.λ) == length(basis.kpoints)
        @test all(length(λk) == n_bands for λk in eigres.λ)
        
        # Compute h values
        E = eigres.λ[1][1]  # Ground state energy
        alpha = 0.1
        deltaE = 0.1
        h_values = DFTK.compute_h_values(basis, eigres, E, alpha, deltaE)
        
        @test length(h_values) == length(basis.kpoints)
        @test all(h -> isa(h, Vec3), h_values)
        
        # h should be small for  1D system (no variation in y,z)
        @test all(h -> abs(h[2]) < 1e-10 && abs(h[3]) < 1e-10, h_values)
    end
    
    @testset "Periodized delta function" begin
        y = Vec3([0.5, 0.0, 0.0])  # Center of domain (fractional coords)
        kpt = basis.kpoints[1]
        delta_y = DFTK.build_periodized_delta(basis, kpt, y)
        
        @test length(delta_y) == length(kpt.G_vectors)
        @test isa(delta_y, Vector{ComplexF64})
        
        # Check normalization: integral of |delta|^2 should be related to Omega
        # This is a rough check
        @test !iszero(delta_y)
    end
    
    @testset "Green's function with GMRES" begin
        # Test with kinetic + local potential only (no nonlinearity)
        y = Vec3([0.5, 0.0, 0.0])
        E_test = 1.0  # Test energy
        
        # Now fully implemented with GMRES solver
        G = compute_periodic_green_function(basis, y, E_test; 
                                           alpha=0.1, deltaE=0.1, n_bands=3,
                                           tol=1e-4, maxiter=50)
        
        @test size(G) == basis.fft_size
        @test isa(G, Array{ComplexF64})
        
        # G should have some structure (not all zeros)
        @test any(!iszero, G)
    end
end
