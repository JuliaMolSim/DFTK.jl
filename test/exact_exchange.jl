@testitem "LiH Hartree-Fock energy" setup=[TestCases, RunSCF] begin
    using DFTK
    using LinearAlgebra
    using PseudoPotentialData

    pd_pbe_family = PseudoFamily("dojo.nc.sr.pbe.v0_5.stringent.upf") 
    Li = ElementPsp(:Li, pd_pbe_family)
    H = ElementPsp(:H, pd_pbe_family)
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

    # regression test
    ref_hf = [[-2.176845488693204, -2.1753394031634743, -2.175339403160776, -2.1753394031568734, 
               -0.374674555189293, -0.1456954541666593, -0.145695454150622, -0.1456954541354867,
                0.309715131365165,  0.3097151313665529,  0.309715131367800]]
    ref_etot = -31.195662141532114
    
    T = Float64
    model  = model_HF(lattice, atoms, positions; 
                      coulomb_kernel_model=WignerSeitzTruncated(), 
                      exx_strategy=CanonicalEXX())
    basis  = PlaneWaveBasis(model, Ecut=40; kgrid=[1, 1, 1])
    
    RunSCF.run_scf_and_compare(T, basis, ref_hf, ref_etot; scf_ene_tol=1e-7, test_tol=5e-5,
                               n_ignored=3, solver=DFTK.scf_damping_solver(damping=1.0),
                               diagtolalg=DFTK.AdaptiveDiagtol(; ratio_ρdiff=5e-4))
end


@testitem "AFM H chain Hartree-Fock energy" setup=[TestCases, RunSCF] begin
    using DFTK
    using LinearAlgebra
    using PseudoPotentialData

    pd_pbe_family = PseudoFamily("dojo.nc.sr.pbe.v0_5.stringent.upf") 
    H = ElementPsp(:H, pd_pbe_family)
    atoms = [H, H, H, H]
    lattice = [[ 20.0  0.0  0.0]; 
               [  0.0  9.0  0.0];
               [  0.0  0.0  9.0]]
    positions = [[0.00, 0.00, 0.00],
                 [0.25, 0.00, 0.00],
                 [0.50, 0.00, 0.00],
                 [0.75, 0.00, 0.00]] 
    Ecut = 32

    # regression test
    ref_hf = [[-0.49538803123060876, -0.4932861126997095, 0.025491794511334334, 0.044421692953926735, 
                0.14646195604973197, 0.16123364389288175, 0.20516020567192236], 
              [-0.49538802813513627, -0.49328610986344545, 0.025491788261921104,
                0.044421685124571855, 0.14646195249587704, 0.16123363562017207, 0.20516020567192264]]
    ref_etot=-2.023997562144
    
    T = Float64
    magnetic_moments = [+1.0, -1.0, +1.0, -1.0]
    model  = model_PBE(lattice, atoms, positions; magnetic_moments, temperature=0.01)
    basis  = PlaneWaveBasis(model; Ecut=Ecut, kgrid=[1, 1, 1])
    ρ = guess_density(basis, magnetic_moments)
    scfres_pbe = self_consistent_field(basis; ρ, is_converged=ScfConvergenceEnergy(1e-8))
    
    model  = model_HF(lattice, atoms, positions; 
                      coulomb_kernel_model=ProbeCharge(), 
                      exx_strategy=ACEXX(),
                      magnetic_moments,
                      temperature=0.01)
    basis  = PlaneWaveBasis(model, Ecut=Ecut; kgrid=[1, 1, 1])
    RunSCF.run_scf_and_compare(T, basis, ref_hf, ref_etot; scf_ene_tol=1e-10, test_tol=5e-5,
                               n_ignored=5, solver=DFTK.scf_damping_solver(damping=0.5),
                               diagtolalg=DFTK.AdaptiveDiagtol(; ratio_ρdiff=1e-6),
                               miniter=20,
                               ρ=scfres_pbe.ρ, 
                               ψ=scfres_pbe.ψ,
                               eigenvalues=scfres_pbe.eigenvalues)
end
