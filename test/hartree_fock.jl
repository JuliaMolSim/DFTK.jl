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
    ref_εF =  0.565145985516737
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
    scfres_pbe = self_consistent_field(basis_pbe; tol=1e-4, seed=0xadcdb6c21c47beb1)

    # Then run Hartree-Fock
    model = model_HF(lattice, atoms, positions;
                     temperature=0.001, smearing=DFTK.Smearing.Gaussian(),
                     exx_algorithm=VanillaExx(),
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

    # TODO: This test is very brittle. I think QE converged to the wrong SCF minimum
end


#
# Regression tests on HF against DFTK itself (TODO: Remove these once we have better tests here)
#
@testitem "LiH Hartree-Fock energy" tags=[:exx,:slow, :dont_test_mpi] setup=[RunSCF] begin
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

    # This created using the very first EXX implementation in DFTK
    ref_hf = [[-2.176845488693204, -2.1753394031634743, -2.175339403160776, -2.1753394031568734, 
               -0.374674555189293, -0.1456954541666593, -0.145695454150622, -0.1456954541354867,
                0.309715131365165,  0.3097151313665529,  0.309715131367800]]
    ref_etot = -31.195662141532114
    
    model  = model_HF(lattice, atoms, positions; 
                      coulomb_kernel_model=WignerSeitzTruncated(), 
                      exx_algorithm=VanillaExx())
    basis  = PlaneWaveBasis(model, Ecut=40; kgrid=[1, 1, 1])
    
    RunSCF.run_scf_and_compare(Float64, basis, ref_hf, ref_etot;
                               scf_ene_tol=1e-10, test_tol=5e-5, n_ignored=0,
                               # TODO: Anderson right does not yet work well for Hartree-Fock
                               damping=0.4, solver=DFTK.scf_damping_solver(),
                               # TODO: The default diagtolalg does not yet work well for Hartree-Fock
                               diagtolalg=DFTK.AdaptiveDiagtol(; ratio_ρdiff=1e-5))
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

    # This created using the very first EXX implementation in DFTK
    ref_hf = [[-2.174882010778448, -2.174882010778414, -2.1748820107783646, -2.1735162108610098,
               -0.4105286062295621, -0.1498412274416261, -0.14984122744054515,
               -0.1498412274386093, 0.39476442887789986, 0.3947644288779635,
               0.39476442887837615]]
    ref_etot = -31.240766149174128

    model  = model_HF(lattice, atoms, positions; 
                      coulomb_kernel_model=SphericallyTruncated(), 
                      exx_algorithm=AceExx())
    basis  = PlaneWaveBasis(model, Ecut=40; kgrid=[1, 1, 1])
    
    RunSCF.run_scf_and_compare(Float64, basis, ref_hf, ref_etot;
                               scf_ene_tol=1e-10, test_tol=5e-5, n_ignored=0,
                               # TODO: Anderson right does not yet work well for Hartree-Fock
                               damping=0.4, solver=DFTK.scf_damping_solver(),
                               # TODO: The default diagtolalg does not yet work well for Hartree-Fock
                               diagtolalg=DFTK.AdaptiveDiagtol(; ratio_ρdiff=1e-5))
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
    Ecut = 32

    # This created using the very first EXX implementation in DFTK
    ref_hf = [[-0.49538803123060876, -0.4932861126997095, 0.025491794511334334, 0.044421692953926735, 
                0.14646195604973197, 0.16123364389288175, 0.20516020567192236], 
              [-0.49538802813513627, -0.49328610986344545, 0.025491788261921104,
                0.044421685124571855, 0.14646195249587704, 0.16123363562017207, 0.20516020567192264]]
    ref_etot=-2.023997562144
    
    magnetic_moments = [+1.0, -1.0, +1.0, -1.0]
    model  = model_DFT(lattice, atoms, positions;
                       magnetic_moments, temperature=0.01, functionals=PBE())
    basis  = PlaneWaveBasis(model; Ecut, kgrid=[1, 1, 1])
    ρ = guess_density(basis, magnetic_moments)
    scfres_pbe = self_consistent_field(basis; ρ, tol=1e-3)
    
    model  = model_HF(lattice, atoms, positions; 
                      coulomb_kernel_model=ProbeCharge(), 
                      exx_algorithm=AceExx(),
                      magnetic_moments,
                      temperature=0.01)
    basis  = PlaneWaveBasis(model; Ecut, kgrid=[1, 1, 1])
    RunSCF.run_scf_and_compare(Float64, basis, ref_hf, ref_etot;
                               scf_ene_tol=1e-10, test_tol=5e-5, n_ignored=0,
                               scfres_pbe.ρ, scfres_pbe.ψ, scfres_pbe.eigenvalues, scfres_pbe.occupation,
                               # TODO: Anderson right does not yet work well for Hartree-Fock
                               damping=0.4, solver=DFTK.scf_damping_solver(),
                               # TODO: The default diagtolalg does not yet work well for Hartree-Fock
                               diagtolalg=DFTK.AdaptiveDiagtol(; ratio_ρdiff=1e-5))
end
