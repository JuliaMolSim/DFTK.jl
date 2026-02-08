@testitem "Phonon: Symmetry transformation of dynamical matrix" #=
    =#    tags=[:phonon, :dont_test_mpi] setup=[Phonon, PhononEwald, TestCases] begin
    using DFTK
    using DFTK: apply_symop_dynmat, normalize_kpoint_coordinate
    using LinearAlgebra

    # Test with a simple cubic system (aluminum is close enough)
    lattice = TestCases.aluminum.lattice
    atoms = TestCases.aluminum.atoms
    positions = TestCases.aluminum.positions
    
    # Create model with symmetries
    model = model_DFT(lattice, atoms, positions; functionals=LDA())
    basis = PlaneWaveBasis(model; Ecut=5, kgrid=[2, 2, 2])
    
    # Get symmetries
    symmetries = basis.symmetries
    @test length(symmetries) > 1  # Should have non-trivial symmetries
    
    # Run SCF
    scfres = self_consistent_field(basis; tol=1e-6, is_converged=ScfConvergenceDensity(1e-6))
    
    # Choose a q-point not at high symmetry (e.g., not Γ, not at zone boundary)
    q = Vec3([0.25, 0.25, 0.0])
    
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
    
    if !isnothing(symop_test)
        println("Testing symmetry transformation: q = $q -> Sq = $q_symm")
        
        # Compute dynamical matrix at q
        dynmat_q = DFTK.compute_dynmat(basis, scfres.ψ, scfres.occupation; 
                                       q, scfres.ρ, scfres.ham, 
                                       scfres.εF, scfres.eigenvalues, tol=1e-8)
        
        # Compute dynamical matrix at Sq directly
        dynmat_Sq_direct = DFTK.compute_dynmat(basis, scfres.ψ, scfres.occupation;
                                               q=q_symm, scfres.ρ, scfres.ham,
                                               scfres.εF, scfres.eigenvalues, tol=1e-8)
        
        # Apply symmetry transformation to get dynamical matrix at Sq
        dynmat_Sq_symm = apply_symop_dynmat(symop_test, model, dynmat_q, q)
        
        # The two should be approximately equal
        # Convert to Cartesian for comparison (as that's the physically meaningful quantity)
        dynmat_Sq_direct_cart = DFTK.dynmat_red_to_cart(model, dynmat_Sq_direct)
        dynmat_Sq_symm_cart = DFTK.dynmat_red_to_cart(model, dynmat_Sq_symm)
        
        # Test that they're close
        @test dynmat_Sq_direct_cart ≈ dynmat_Sq_symm_cart rtol=1e-5
        
        # Also check that phonon frequencies are the same
        modes_direct = DFTK._phonon_modes(basis, dynmat_Sq_direct_cart)
        modes_symm = DFTK._phonon_modes(basis, dynmat_Sq_symm_cart)
        
        @test modes_direct.frequencies ≈ modes_symm.frequencies rtol=1e-5
    else
        @info "No suitable symmetry operation found for testing (all symmetries preserve q = $q)"
    end
end

@testitem "Phonon: Irreducible q-point reduction" #=
    =#    tags=[:phonon, :dont_test_mpi] setup=[TestCases] begin
    using DFTK
    using DFTK: get_irreducible_qpoints, normalize_kpoint_coordinate
    
    # Test with aluminum (FCC has good symmetry)
    lattice = TestCases.aluminum.lattice  
    atoms = TestCases.aluminum.atoms
    positions = TestCases.aluminum.positions
    
    model = model_DFT(lattice, atoms, positions; functionals=LDA())
    basis = PlaneWaveBasis(model; Ecut=5, kgrid=[2, 2, 2])
    
    symmetries = basis.symmetries
    
    # Create a simple grid of q-points
    qpoints = [Vec3([i/4, j/4, k/4]) for i in 0:3, j in 0:3, k in 0:3] |> vec
    qpoints = normalize_kpoint_coordinate.(qpoints)
    
    # Get irreducible q-points
    result = get_irreducible_qpoints(qpoints, symmetries)
    
    @test length(result.qpoints_irred) < length(qpoints)  # Should reduce
    @test length(result.mapping) == length(qpoints)
    
    # Check that all q-points can be recovered from irreducible ones
    for (iq, q) in enumerate(qpoints)
        iq_irred, symop = result.mapping[iq]
        q_irred = result.qpoints_irred[iq_irred]
        q_recovered = normalize_kpoint_coordinate(symop.S * q_irred)
        @test norm(q_recovered - q) < 1e-8
    end
end
