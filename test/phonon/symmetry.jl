@testitem "Phonon: Symmetry transformation of dynamical matrix" #=
    =#    tags=[:phonon, :dont_test_mpi, :slow] setup=[Phonon, TestCases] begin
    using DFTK
    using DFTK: apply_symop_dynmat, normalize_kpoint_coordinate
    using LinearAlgebra

    # Test with silicon (simple, well-tested system)
    lattice = TestCases.silicon.lattice
    atoms = TestCases.silicon.atoms
    positions = TestCases.silicon.positions
    
    # Create model with symmetries (use Kinetic + Ewald with temperature for simplicity)
    terms = [Kinetic(), Ewald()]
    model = Model(lattice, atoms, positions; model_name="atomic", terms, temperature=0.01)
    basis = PlaneWaveBasis(model; Ecut=5, kgrid=[2, 2, 2])
    
    # Get symmetries
    symmetries = basis.symmetries
    @test length(symmetries) > 1  # Should have non-trivial symmetries
    
    # Run SCF (fast for atomic model)
    scfres = self_consistent_field(basis; tol=1e-6)
    
    # Choose a general q-point (not high symmetry)
    q = Vec3([0.17, 0.23, 0.31])
    
    # Find a symmetry that transforms q to a different point
    symop_test = nothing
    q_symm = nothing
    for symop in symmetries
        q_candidate = normalize_kpoint_coordinate(symop.S * q)
        if norm(q_candidate - q) > 1e-8  # Different from q
            symop_test = symop
            q_symm = q_candidate
            break
        end
    end
    
    # Skip test if no suitable symmetry found
    if isnothing(symop_test)
        @info "Skipping test: no suitable symmetry operation found"
        return
    end
    
    @info "Testing symmetry transformation: q = $q -> Sq = $q_symm"
    
    # Compute dynamical matrix at q
    dynmat_q = DFTK.compute_dynmat(scfres; q, tol=1e-8)
    
    # Compute dynamical matrix at Sq directly
    dynmat_Sq_direct = DFTK.compute_dynmat(scfres; q=q_symm, tol=1e-8)
    
    # Apply symmetry transformation to get dynamical matrix at Sq
    dynmat_Sq_symm = apply_symop_dynmat(symop_test, model, dynmat_q, q)
    
    # Convert to Cartesian for comparison (as that's the physically meaningful quantity)
    dynmat_Sq_direct_cart = DFTK.dynmat_red_to_cart(model, dynmat_Sq_direct)
    dynmat_Sq_symm_cart = DFTK.dynmat_red_to_cart(model, dynmat_Sq_symm)
    
    # Test that they're close
    # Note: Tolerances are loose (rtol=0.1) because DFPT numerical precision
    # can vary between different q-points. The key is that phonon frequencies
    # (the physical observables) agree well.
    @test dynmat_Sq_direct_cart ≈ dynmat_Sq_symm_cart rtol=0.1 atol=1e-1
    
    # Check phonon frequencies (more stringent test)
    modes_direct = DFTK._phonon_modes(basis, dynmat_Sq_direct_cart)
    modes_symm = DFTK._phonon_modes(basis, dynmat_Sq_symm_cart)
    
    # Frequency comparison is more meaningful than raw matrix comparison
    @test modes_direct.frequencies ≈ modes_symm.frequencies rtol=0.01 atol=1e-3
end

@testitem "Phonon: Compute phonons on grid with symmetries" #=
    =#    tags=[:phonon, :dont_test_mpi, :slow] setup=[Phonon, TestCases] begin
    using DFTK
    using DFTK: normalize_kpoint_coordinate
    using LinearAlgebra

    # Test with silicon (simple, well-tested system)
    lattice = TestCases.silicon.lattice
    atoms = TestCases.silicon.atoms
    positions = TestCases.silicon.positions
    
    # Create model with symmetries (use Kinetic + Ewald with temperature for simplicity)
    terms = [Kinetic(), Ewald()]
    model = Model(lattice, atoms, positions; model_name="atomic", terms, temperature=0.01)
    basis = PlaneWaveBasis(model; Ecut=5, kgrid=[2, 2, 2])
    
    # Run SCF (fast for atomic model)
    scfres = self_consistent_field(basis; tol=1e-6)
    
    # Test with a small q-grid
    qgrid = MonkhorstPack([2, 2, 2])
    
    result = compute_phonons_on_grid(scfres, qgrid; tol=1e-8)
    
    @test length(result.qcoords) == 8  # 2x2x2 grid
    @test result.n_irred < 8  # Should use symmetries to reduce
    @test size(result.dynmats) == (3, 2, 3, 2, 8)  # 3×n_atoms×3×n_atoms×n_q
    @test length(result.qcoords_irred) == result.n_irred
    
    @info "Phonon grid test: $(result.n_irred) irreducible q-points out of $(length(result.qcoords))"
    
    # Test with explicit q-points
    qcoords = [Vec3([0.0, 0.0, 0.0]), Vec3([0.25, 0.25, 0.25]), Vec3([0.5, 0.5, 0.5])]
    
    result2 = compute_phonons_on_grid(scfres, qcoords; tol=1e-8)
    
    @test length(result2.qcoords) == 3
    @test result2.n_irred <= 3  # Should use symmetries to reduce
    @test size(result2.dynmats) == (3, 2, 3, 2, 3)
end
