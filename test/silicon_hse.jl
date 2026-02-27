@testitem "Silicon HSE" tags=[:minimal, :exx, :dont_test_mpi] setup=[RunSCF] begin
    using DFTK
    using PseudoPotentialData
    using .RunSCF: run_scf_and_compare

    # These values were computed using QuantumEspresso with one kpoint and Ecut = 25
    # using exactly the same settings (no ACE)
    #
    ref_εF = 2.637577380644947E-001
    ref_hf = [
        [-3.966099538728215E-001, 6.079406112827507E-002,  6.079406112846138E-002,
          6.079406112862479E-002, 3.816652905730460E-001,  3.816652905730735E-001,
          3.816652905730933E-001, 4.326487809357601E-001],
    ]
    ref_etot = -8.398986637133419E+000

    # Adjust bands to Fermi level changes between QE and DFTK
    δεF = 0.19455408549950615
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
    basis_pbe  = PlaneWaveBasis(model_pbe; Ecut=25, kgrid=[1, 1, 1])
    scfres_pbe = self_consistent_field(basis_pbe; tol=1e-4)

    hse = HSE(exx_algorithm=VanillaExx())
    model = model_DFT(lattice, atoms, positions;
                      temperature=0.001, smearing=DFTK.Smearing.Gaussian(),
                      functionals=hse)
    basis = PlaneWaveBasis(model; Ecut=25, kgrid=[1, 1, 1])

    # Note: With ACE enabled, the unoccupied orbitals are represented rather poorly.
    run_scf_and_compare(Float64, basis, ref_hf, ref_etot; 
                        scf_ene_tol=1e-8, test_tol=1e-4, n_ignored=4,
                        scfres_pbe.ψ, scfres_pbe.ρ,
                        scfres_pbe.eigenvalues, scfres_pbe.occupation,
                        # TODO: Anderson right does not yet work well for Hartree-Fock
                        damping=0.3, solver=DFTK.scf_damping_solver(),
                        # TODO: The default diagtolalg does not yet work well for Hartree-Fock
                        diagtolalg=DFTK.AdaptiveDiagtol(; ratio_ρdiff=1e-5))
end
