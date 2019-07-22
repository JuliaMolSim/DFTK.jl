using Test
using DFTK
using Libxc: Functional

include("silicon_testcases.jl")

function run_silicon_noXC(;Ecut=5, test_tol=1e-6, n_ignored=0, grid_size=15, scf_tol=1e-6)
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
    Si = Species(atnum, psp=load_psp("si-pade-q4.hgh"))
    n_electrons = length(positions) * n_elec_valence(Si)

    # Construct a Hamiltonian (Kinetic + local psp + nonlocal psp + Hartree)
    ham = Hamiltonian(basis, pot_local=build_local_potential(basis, Si => positions),
                      pot_nonlocal=build_nonlocal_projectors(basis, Si => positions),
                      pot_hartree=PotHartree(basis))

    ρ = guess_gaussian_sad(basis, Si => positions)
    prec = PreconditionerKinetic(ham, α=0.1)
    scfres = self_consistent_field(ham, n_bands, n_electrons, lobpcg_prec=prec, ρ=ρ,
                                   tol=scf_tol)

    for ik in 1:length(kpoints)
        println(ik, "  ", abs.(ref_noXC[ik] - scfres.orben[ik]))
    end
    for ik in 1:length(kpoints)
        # Ignore last few bands, because these eigenvalues are hardest to converge
        # and typically a bit random and unstable in the LOBPCG
        diff = abs.(ref_noXC[ik] - scfres.orben[ik])
        @test maximum(diff[1:n_bands - n_ignored]) < test_tol
    end
end

function run_silicon_lda(T ;Ecut=5, test_tol=1e-6, n_ignored=0, grid_size=15, scf_tol=1e-6)
    # These values were computed using
    # mfherbst/PlaneWaveToyPrograms.jl/2019.03_Simple_DFT/test_Bandstructure.jl with Ecut = 25
    ref_lda = [
        [-0.178051301635369, 0.261398506992241, 0.262576260326963, 0.263078571863638,
          0.352988658592485, 0.354755849352724, 0.355790898546273, 0.377652662160092,
          0.540863835768829, 0.545642032223014],
        [-0.127609438354543, 0.066093170654560, 0.225532280043458, 0.226147707206294,
          0.319771354243122, 0.387035289210488, 0.389045392551022, 0.541640786469015,
          0.555193302652323, 0.555438444446987],
        [-0.107969256184491, 0.077436393756951, 0.173232116228659, 0.173447048182548,
          0.283675990765947, 0.329904374121225, 0.525869414337404, 0.527031259293798,
          0.613354356913113, 0.625702042941051],
        [-0.058393554734870, 0.014066393945345, 0.098114979575642, 0.184797179474013,
          0.312283152046114, 0.472042031771217, 0.498745505023607, 0.517542258623193,
          0.528431528397109, 0.542724927211605],
    ]
    n_bands = length(ref_lda[1])

    basis = PlaneWaveBasis(Array{T}(lattice), grid_size * ones(3), Ecut, kpoints, kweights)
    Si = Species(atnum, psp=load_psp("si-pade-q4.hgh"))
    n_electrons = length(positions) * n_elec_valence(Si)

    # Construct the Hamiltonian
    ham = Hamiltonian(basis, pot_local=build_local_potential(basis, Si => positions),
                      pot_nonlocal=build_nonlocal_projectors(basis, Si => positions),
                      pot_hartree=PotHartree(basis),
                      pot_xc=PotXc(basis, :lda_x, :lda_c_vwn))

    # Construct guess and run the SCF
    ρ = guess_gaussian_sad(basis, Si => positions)
    prec = PreconditionerKinetic(ham, α=0.1)
    scfres = self_consistent_field(ham, n_bands, n_electrons, lobpcg_prec=prec, ρ=ρ,
                                   tol=scf_tol)

    for ik in 1:length(kpoints)
        @test eltype(scfres.orben[ik]) == T
        @test eltype(scfres.Psi[ik]) == Complex{T}
        println(ik, "  ", abs.(ref_lda[ik] - scfres.orben[ik]))
    end
    for ik in 1:length(kpoints)
        # Ignore last few bands, because these eigenvalues are hardest to converge
        # and typically a bit random and unstable in the LOBPCG
        diff = abs.(ref_lda[ik] - scfres.orben[ik])
        @test maximum(diff[1:n_bands - n_ignored]) < test_tol
    end
end
