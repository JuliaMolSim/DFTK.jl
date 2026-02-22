@testitem "Silicon PBE0" tags=[:minimal, :exx, :dont_test_mpi] setup=[RunSCF] begin
    using DFTK
    using PseudoPotentialData
    using .RunSCF: run_scf_and_compare

    # These values were computed using QuantumEspresso with one kpoint and Ecut = 20
    # using exactly the same settings (ACE, no treatment of Coulomb singularity)
    #
    ref_εF = 0.3890179653333848
    ref_hf = [
        [-1.423449409602823E-001, 3.155230816687672E-001, 3.155230816689635E-001,
          3.155230816693947E-001, 3.948598418578042E-001, 3.948598418580224E-001,
          3.948598418581931E-001, 4.459496545492150E-001],
    ]
    ref_etot = -7.299027688178781

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
    scfres_pbe = self_consistent_field(basis_pbe; tol=1e-4, seed=0xadcdb6c21c47beb1)

    # Then run Hartree-Fock
    model = model_HF(lattice, atoms, positions;
                     temperature=0.001, smearing=DFTK.Smearing.Gaussian(),
                     exx_algorithm=AceExx(),
                     coulomb_kernel_model=NeglectSingularity())
    basis = PlaneWaveBasis(model; Ecut=20, kgrid=[1, 1, 1])

    run_scf_and_compare(Float64, basis, ref_hf, ref_etot; 
                        scf_ene_tol=1e-7, test_tol=1e-4, maxiter=20,
                        scfres_pbe.ψ, scfres_pbe.ρ,
                        scfres_pbe.eigenvalues, scfres_pbe.occupation,
                        # TODO: Anderson right does not yet work well for Hartree-Fock
                        damping=0.3, solver=DFTK.scf_damping_solver(),
                        # TODO: The default diagtolalg does not yet work well for Hartree-Fock
                        diagtolalg=DFTK.AdaptiveDiagtol(; ratio_ρdiff=1e-5))
end
