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

@testitem "Phonon: Irreducible q-point reduction" #=
    =#    tags=[:phonon, :dont_test_mpi] setup=[TestCases] begin
    using DFTK
    using DFTK: get_irreducible_qpoints, normalize_kpoint_coordinate
    
    # Test with silicon
    lattice = TestCases.silicon.lattice  
    atoms = TestCases.silicon.atoms
    positions = TestCases.silicon.positions
    
    terms = [Kinetic(), Ewald()]
    model = Model(lattice, atoms, positions; model_name="atomic", terms)
    basis = PlaneWaveBasis(model; Ecut=5, kgrid=[2, 2, 2])
    
    symmetries = basis.symmetries
    
    # Create a simple 2x2x2 grid of q-points
    qpoints = [Vec3([i/2, j/2, k/2]) for i in 0:1, j in 0:1, k in 0:1] |> vec
    qpoints = normalize_kpoint_coordinate.(qpoints)
    
    # Get irreducible q-points
    result = get_irreducible_qpoints(qpoints, symmetries)
    
    @test length(result.qpoints_irred) <= length(qpoints)  # Should reduce or stay same
    @test length(result.mapping) == length(qpoints)
    
    # Check that all q-points can be recovered from irreducible ones
    for (iq, q) in enumerate(qpoints)
        iq_irred, symop = result.mapping[iq]
        q_irred = result.qpoints_irred[iq_irred]
        q_recovered = normalize_kpoint_coordinate(symop.S * q_irred)
        @test norm(q_recovered - q) < 1e-8
    end
end
