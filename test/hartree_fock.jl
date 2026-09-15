#
# Test the most simplistic HF model on one k-point against QuantumEspresso reference data
#
@testitem "Silicon Hartree-Fock" tags=[:minimal, :exx, :dont_test_mpi] setup=[RunSCF] begin
    using DFTK
    using PseudoPotentialData
    using .RunSCF: run_scf_and_compare

    # These values were computed using QuantumEspresso with one kpoint and Ecut = 20
    # using exactly the same settings (no ACE, no treatment of Coulomb singularity)
    #
    ref_εF = 0.565145985516737
    ref_hf = [
        [2.833325458164758E-002, 5.122487481436300E-001, 5.122487481437534E-001,
         5.122487481437670E-001, 5.880655155002166E-001, 5.880655155002978E-001,
         5.880655155003645E-001, 6.642767195301558E-001],
    ]
    ref_etot = -5.344212801278319

    # Adjust bands to Fermi level changes between QE and DFTK
    δεF = 0.49594239176094174
    ref_hf = [e .+ (- ref_εF + δεF) for e in ref_hf]

    # First run PBE to get initial guess
    a = 5.13
    lattice = a * [[0 1 1.];
                   [1 0 1.];
                   [1 1 0.]]
    positions = [ones(3)/8, -ones(3)/8]

    pseudopotentials = PseudoFamily("dojo.nc.sr.pbe.v0_4_1.standard.upf")
    Si = ElementPsp(:Si, pseudopotentials)
    atoms = [Si, Si]

    model_pbe  = model_DFT(lattice, atoms, positions;
                           functionals=PBE(), temperature=0.001,
                           smearing=DFTK.Smearing.Gaussian())
    basis_pbe  = PlaneWaveBasis(model_pbe; Ecut=20, kgrid=[1, 1, 1])
    scfres_pbe = self_consistent_field(basis_pbe; callback=identity,
                                       tol=1e-4, seed=0xadcdb6c21c47beb1)

    # Then run Hartree-Fock
    model = model_HF(lattice, atoms, positions;
                     temperature=0.001, smearing=DFTK.Smearing.Gaussian(),
                     exx_kernel=Coulomb(ReplaceSingularity(0.0)))
    basis = PlaneWaveBasis(model; Ecut=20, kgrid=[1, 1, 1])

    run_scf_and_compare(Float64, basis, ref_hf, ref_etot; 
                        scf_dens_tol=5e-4, test_tol=1e-4, n_ignored=1, maxiter=30,
                        scfres_pbe.ψ, scfres_pbe.ρ,
                        scfres_pbe.eigenvalues, scfres_pbe.occupation, exxalg=VanillaExx(),
                        # TODO: Anderson right does not yet work well for Hartree-Fock
                        damping=0.3, solver=DFTK.scf_damping_solver())

    # TODO: This test is very brittle. I think QE converged to the wrong SCF minimum
end

@testitem "Hartree-Fock k-point consistency" tags=[:exx, :dont_test_mpi] setup=[TestCases] begin
    using DFTK
    using .TestCases: silicon

    # Comparison of a 3x2x1 k-grid with a 3x2x1 supercell at Gamma
    # using ProbeCharge and AceExx.
    Ecut  = 5
    kgrid = [3, 2, 1]
    Si    = ElementPsp(silicon.atnum, load_psp(silicon.psp_upf))
    
    # We use a temperature to ensure unique occupations and better convergence
    model = model_HF(silicon.lattice, [Si, Si], silicon.positions;
                     exx_kernel=Coulomb(ProbeCharge()),
                     temperature=1e-3, smearing=DFTK.Smearing.Gaussian())
    basis = PlaneWaveBasis(model; Ecut, kgrid)

    # 1. Unit cell calculation
    scfres = self_consistent_field(basis; is_converged=ScfConvergenceEnergy(1e-10), 
                                   exxalg=AceExx(), solver=DFTK.scf_damping_solver(), damping=0.4)

    # 2. Supercell calculation
    basis_supercell = cell_to_supercell(basis)
    scfres_supercell = self_consistent_field(basis_supercell; is_converged=ScfConvergenceEnergy(1e-10),
                                             exxalg=AceExx(), solver=DFTK.scf_damping_solver(), 
                                             damping=0.4)

    # Energy per unit cell should be the same
    @test scfres.energies.total * prod(kgrid) ≈ scfres_supercell.energies.total atol=1e-6
end

@testitem "Hartree-Fock collinear spin without magnetisation (k-points)" #=
        =# tags=[:exx, :dont_test_mpi] setup=[TestCases] begin
    using DFTK
    using LinearAlgebra
    using .TestCases: silicon

    # A collinear calculation with identical spin-up and spin-down orbitals and halved
    # occupations has to reproduce the spin-unpolarised exchange energy and Hamiltonian.
    # This checks the spin bookkeeping of the exchange term in combination with k-points
    # (occupation convention, k-point weights, spin-aware q-point mapping).
    Ecut    = 5
    kgrid   = MonkhorstPack([2, 1, 2]; kshift=[1/2, 0, 1/2])
    n_bands = 6
    Si      = ElementPsp(silicon.atnum, load_psp(silicon.psp_upf))

    function make_basis(spin_polarization)
        model = Model(silicon.lattice, [Si, Si], silicon.positions; spin_polarization,
                      terms=[ExactExchange(; kernel=Coulomb(ProbeCharge()))])
        PlaneWaveBasis(model; Ecut, kgrid)
    end
    basis     = make_basis(:none)
    basis_col = make_basis(:collinear)
    n_k = length(basis.kpoints)
    @test length(basis_col.kpoints) == 2n_k

    ψ = [Matrix(qr(randn(ComplexF64, length(G_vectors(basis, kpt)), n_bands)).Q)
         for kpt in basis.kpoints]
    occupation = [[2.0, 2.0, 2.0, 1.2, 0.8, 0.0] for _ = 1:n_k]  # filled_occupation = 2

    # Same orbitals in both spin channels, occupations halved (filled_occupation = 1)
    ψ_col   = vcat(ψ, ψ)
    occ_col = [occk ./ 2 for occk in vcat(occupation, occupation)]
    for ik = 1:n_k, σ = 1:2
        ik_col = DFTK.krange_spin(basis_col, σ)[ik]
        @test basis_col.kpoints[ik_col].coordinate == basis.kpoints[ik].coordinate
    end

    res     = energy_hamiltonian(basis,     ψ,     occupation; exxalg=VanillaExx())
    res_col = energy_hamiltonian(basis_col, ψ_col, occ_col;    exxalg=VanillaExx())
    @test res.energies.total ≈ res_col.energies.total rtol=1e-10

    for ik = 1:n_k
        Hψk = res.ham.blocks[ik] * ψ[ik]
        for σ = 1:2
            ik_col = DFTK.krange_spin(basis_col, σ)[ik]
            @test Hψk ≈ res_col.ham.blocks[ik_col] * ψ_col[ik_col] rtol=1e-10
        end
    end
end

@testitem "AFM H chain Hartree-Fock k-point consistency" #=
        =# tags=[:exx, :dont_test_mpi] begin
    using DFTK
    using LinearAlgebra
    using PseudoPotentialData

    # Comparison of an antiferromagnetic H chain with a 2x1x1 k-grid against the
    # 2x1x1 supercell at Gamma. This checks the spin-matched k' = k - q lookup
    # of the exchange term in a case where the spin channels genuinely differ.
    pseudopotentials = PseudoFamily("dojo.nc.sr.pbe.v0_5.stringent.upf")
    H = ElementPsp(:H, pseudopotentials)
    lattice   = diagm([10.0, 8.0, 8.0])
    positions = [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]]
    magnetic_moments = [+1.0, -1.0]
    system = DFTK.periodic_system(lattice, [H, H], positions)
    Ecut  = 10
    kgrid = [2, 1, 1]

    model_pbe = model_DFT(system; pseudopotentials, magnetic_moments, temperature=0.01,
                          functionals=PBE())
    model_hf  = model_HF(system; pseudopotentials, magnetic_moments, temperature=0.01,
                         exx_kernel=Coulomb(ProbeCharge()))
    basis_pbe = PlaneWaveBasis(model_pbe; Ecut, kgrid)
    basis_hf  = PlaneWaveBasis(model_hf;  Ecut, kgrid)

    # Start Hartree-Fock from a converged PBE state, such that the unit cell and the
    # supercell calculation converge to the same antiferromagnetic solution.
    function run_hf(basis_pbe, basis_hf, magnetic_moments)
        ρ = guess_density(basis_pbe, magnetic_moments)
        scfres_pbe = self_consistent_field(basis_pbe; ρ, tol=1e-6)

        # Sketch ACE with the occupied orbitals only, as the other spin-polarised tests do.
        # With the default sketch (including the extra bands) the Cholesky factorisation in
        # the ACE compression failed for this system when starting from the atomic guess.
        self_consistent_field(basis_hf; scfres_pbe.ρ, scfres_pbe.ψ, scfres_pbe.eigenvalues,
                              scfres_pbe.occupation, is_converged=ScfConvergenceEnergy(1e-10),
                              exxalg=AceExx(sketch_with_extra_orbitals=false),
                              solver=DFTK.scf_damping_solver(), damping=0.4)
    end

    # 1. Unit cell calculation
    scfres = run_hf(basis_pbe, basis_hf, magnetic_moments)
    @test maximum(abs, spin_density(scfres.ρ)) > 1e-2  # state is actually magnetic

    # 2. Supercell calculation (create_supercell lists all images of an atom consecutively)
    magnetic_moments_supercell = repeat(magnetic_moments; inner=prod(kgrid))
    scfres_supercell = run_hf(cell_to_supercell(basis_pbe), cell_to_supercell(basis_hf),
                              magnetic_moments_supercell)

    # Energy per unit cell should be the same
    @test scfres.energies.total * prod(kgrid) ≈ scfres_supercell.energies.total atol=1e-6
end

@testitem "LiH Hartree-Fock energy" tags=[:exx,:slow] setup=[RunSCF] begin
    using DFTK
    using LinearAlgebra
    using PseudoPotentialData

    pseudopotentials = PseudoFamily("dojo.nc.sr.pbe.v0_5.stringent.upf") 
    Li = ElementPsp(:Li, pseudopotentials)
    H  = ElementPsp(:H,  pseudopotentials)
    atoms = [Li, Li, Li, Li, H, H, H, H]
    a = 7.504
    lattice = a * I(3)
    positions = [[0.0, 0.0, 0.0], 
                 [0.5, 0.5, 0.0],
                 [0.0, 0.5, 0.5],
                 [0.5, 0.0, 0.5],
                 [0.5, 0.0, 0.0],
                 [0.0, 0.5, 0.0],
                 [0.0, 0.0, 0.5],
                 [0.5, 0.5, 0.5]]

    # The eigenvalues were created using DFTK
    ref_hf = [[-2.174882010778448, -2.174882010778414, -2.1748820107783646, -2.1735162108610098,
               -0.4105286062295621, -0.1498412274416261, -0.14984122744054515,
               -0.1498412274386093, 0.39476442887789986, 0.3947644288779635,
               0.39476442887837615]]
    ref_etot = -31.241195440836385

    model  = model_HF(lattice, atoms, positions; exx_kernel=SphericallyTruncatedCoulomb())
    basis  = PlaneWaveBasis(model, Ecut=40; kgrid=[1, 1, 1])

    # Note: In DFTK we disable ACE for the final energy computation and the final Hamiltonian
    #       build, why QE does not do that. Hence our total energy agrees only to 1e-4
    RunSCF.run_scf_and_compare(Float64, basis, ref_hf, ref_etot;
                               scf_ene_tol=1e-10, test_tol=1e-4, n_ignored=0,
                               exxalg=AceExx(sketch_with_extra_orbitals=false),
                               # TODO: Anderson right does not yet work well for Hartree-Fock
                               damping=0.4, solver=DFTK.scf_damping_solver())
end

@testitem "AFM H chain Hartree-Fock energy" tags=[:exx, :dont_test_mpi] setup=[RunSCF] begin
    using DFTK
    using LinearAlgebra
    using PseudoPotentialData

    pseudopotentials = PseudoFamily("dojo.nc.sr.pbe.v0_5.stringent.upf") 
    H = ElementPsp(:H, pseudopotentials)
    atoms = [H, H, H, H]
    lattice = [[ 20.0  0.0  0.0]; 
               [  0.0  9.0  0.0];
               [  0.0  0.0  9.0]]
    positions = [[0.00, 0.00, 0.00],
                 [0.25, 0.00, 0.00],
                 [0.50, 0.00, 0.00],
                 [0.75, 0.00, 0.00]] 
    system = DFTK.periodic_system(lattice, atoms, positions)

    Ecut = 32

    # This created using the very first EXX implementation in DFTK
    ref_hf = [[-0.49538803123060876, -0.4932861126997095, 0.025491794511334334, 0.044421692953926735, 
                0.14646195604973197, 0.16123364389288175, 0.20516020567192236], 
              [-0.49538802813513627, -0.49328610986344545, 0.025491788261921104,
                0.044421685124571855, 0.14646195249587704, 0.16123363562017207, 0.20516020567192264]]
    ref_etot=-2.023997562144
    
    magnetic_moments = [+1.0, -1.0, +1.0, -1.0]
    model  = model_DFT(system; pseudopotentials, magnetic_moments,
                       temperature=0.01, functionals=PBE())
    basis  = PlaneWaveBasis(model; Ecut, kgrid=[1, 1, 1])
    ρ = guess_density(basis, magnetic_moments)
    scfres_pbe = self_consistent_field(basis; ρ, tol=1e-3)
    
    model  = model_HF(system; pseudopotentials, magnetic_moments, temperature=0.01,
                      exx_kernel=Coulomb(ProbeCharge()))
    basis  = PlaneWaveBasis(model; Ecut, kgrid=[1, 1, 1])

    RunSCF.run_scf_and_compare(Float64, basis, ref_hf, ref_etot;
                               scf_ene_tol=1e-10, test_tol=5e-5, n_ignored=0,
                               scfres_pbe.ρ, scfres_pbe.ψ, scfres_pbe.eigenvalues,
                               scfres_pbe.occupation,
                               exxalg=AceExx(sketch_with_extra_orbitals=false),
                               # TODO: Anderson right does not yet work well for Hartree-Fock
                               damping=0.4, solver=DFTK.scf_damping_solver())
end
