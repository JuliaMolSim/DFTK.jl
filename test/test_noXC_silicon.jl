using Test
using DFTK
using LinearAlgebra

include("testcases_silicon.jl")

function run_noXC(;Ecut=5, test_tol=1e-6, n_ignored=0, grid_size=15)
    # T + Vloc + Vnloc + Vhartree
    # These values were computed using
    # mfherbst/PlaneWaveToyPrograms.jl/2019.03_Simple_DFT/test_Bandstructure.jl with Ecut = 25
    ref_noXC = [
        [0.185549922088566, 0.639580538964015, 0.646030397885458, 0.652391901810345,
         0.702347771578107, 0.710469043935530, 0.717231516421613, 0.748840642735156,
         0.846286646845372, 0.897861139946736],
        [0.239155789322723, 0.429540733681777, 0.603086037153154, 0.610580188077710,
         0.670337883132119, 0.732324695330319, 0.744702551064525, 0.849154438618379,
         0.905843481782612, 0.913384018361851],
        [0.259218706349742, 0.455276031447196, 0.540592450136986, 0.547611851594001,
         0.614616466461915, 0.656596415752623, 0.891416746968197, 0.901515031975119,
         0.978698738042756, 0.988928267296001],
        [0.314838087322010, 0.387058998640207, 0.447167996979893, 0.561262061172101,
         0.637329096727452, 0.820133740395368, 0.863328896122870, 0.877509936248489,
         0.880721134709112, 0.899689743018489],
    ]

    n_bands = length(ref_noXC[1])
    basis = PlaneWaveBasis(lattice, grid_size * ones(3), Ecut, kpoints, kweights)

    # Construct a local pseudopotential
    hgh = load_psp("si-pade-q4.hgh")
    psp_local = build_local_potential(basis, positions,
                                      G -> DFTK.eval_psp_local_fourier(hgh, basis.recip_lattice * G))
    psp_nonlocal = PotNonLocal(basis, "Si" => positions, "Si" => hgh)
    n_filled = 4  # In a Silicon psp model, the number of electrons per unit cell is 8

    # Construct a Hamiltonian (Kinetic + local psp + nonlocal psp + Hartree)
    ham = Hamiltonian(basis, pot_local=psp_local,
                      pot_nonlocal=psp_nonlocal,
                      pot_hartree=PotHartree(basis))

    prec = PreconditionerKinetic(ham, α=0.1)
    scfres = self_consistent_field(ham, n_filled + 4, n_filled,
                                   lobpcg_prec=prec)
    ρ_Y, pot_hartree_values, pot_xc_values = scfres
    res = lobpcg(ham, n_bands, pot_hartree_values=pot_hartree_values,
                 pot_xc_values=pot_xc_values, prec=prec, tol=1e-6)

    for ik in 1:length(kpoints)
        println(ik, "  ", abs.(ref_noXC[ik] - res.λ[ik]))
    end
    for ik in 1:length(kpoints)
        # Ignore last few bands, because these eigenvalues are hardest to converge
        # and typically a bit random and unstable in the LOBPCG
        diff = abs.(ref_noXC[ik] - res.λ[ik])
        @test maximum(diff[1:n_bands - n_ignored]) < test_tol
    end
end

@testset "SCF of silicon without exchange-correlation (small)" begin
    run_noXC(Ecut=5, test_tol=0.05, n_ignored=0, grid_size=15)
end

@testset "SCF of silicon without exchange-correlation (medium)" begin
    if ! running_in_ci
        run_noXC(Ecut=15, test_tol=0.0005, n_ignored=5, grid_size=25)
    else
        println("Skipping medium test, since running from CI.")
    end
end

@testset "SCF of silicon without exchange-correlation (large)" begin
    if ! running_in_ci
        run_noXC(Ecut=25, test_tol=5e-7, n_ignored=0, grid_size=33)
    else
        println("Skipping large test, since running from CI.")
    end
end
