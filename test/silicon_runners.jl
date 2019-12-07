using Test
using DFTK

include("testcases.jl")

# TODO There is a lot of code duplication in this file ... once we have the ABINIT reference
#      stuff in place, this should be refactored.

function run_silicon_redHF(T; Ecut=5, test_tol=1e-6, n_ignored=0, grid_size=15, scf_tol=1e-6)
    # T + Vloc + Vnloc + Vhartree
    # These values were computed using ABINIT with the same kpoints as testcases.jl
    # and Ecut = 25
    ref_redHF = [
        [0.185624211511768, 0.645877543801093, 0.645877543804049, 0.645877543807010,
         0.710639433524012, 0.710639433527887, 0.710639433531763, 0.747770226622483,
         0.846692796204658, 0.903426057105463],
        [0.239766326671308, 0.428509722981838, 0.603258008732289, 0.603258008735511,
         0.677812317259065, 0.740270198505120, 0.740270198508514, 0.855980714202823,
         0.904920991452396, 0.904920991455659],
        [0.259399404104112, 0.456356401369978, 0.541006959363826, 0.541006959368758,
         0.620449688893274, 0.658935639559930, 0.894581352229539, 0.894581352233318,
         0.982503349423807, 0.982875170859166],
        [0.316557263535848, 0.385639894339023, 0.446179301437491, 0.554693942841539,
         0.649929646792239, 0.820125934482662, 0.856479348217456, 0.880349468106374,
         0.884889516616732, 0.892844865916440],
    ]
    n_bands = length(ref_redHF[1])

    fft_size = grid_size * ones(3)
    Si = Species(silicon.atnum, psp=load_psp(silicon.psp))
    model = model_reduced_hf(Array{T}(silicon.lattice), Si => silicon.positions)
    basis = PlaneWaveBasis(model, Ecut, silicon.kcoords, silicon.ksymops; fft_size=fft_size)
    ham = Hamiltonian(basis, guess_density(basis, Si => silicon.positions))

    scfres = self_consistent_field(ham, n_bands, tol=scf_tol)

    for ik in 1:length(silicon.kcoords)
        @test eltype(scfres.orben[ik]) == T
        @test eltype(scfres.Psi[ik]) == Complex{T}
        println(ik, "  ", abs.(ref_redHF[ik] - scfres.orben[ik]))
    end
    for ik in 1:length(silicon.kcoords)
        # Ignore last few bands, because these eigenvalues are hardest to converge
        # and typically a bit random and unstable in the LOBPCG
        diff = abs.(ref_redHF[ik] - scfres.orben[ik])
        @test maximum(diff[1:n_bands - n_ignored]) < test_tol
    end
end

function run_silicon_lda(T ;Ecut=5, test_tol=1e-6, n_ignored=0, grid_size=15, scf_tol=1e-6,
                         lobpcg_tol=scf_tol / 10, n_noconv_check=0)
    # These values were computed using ABINIT with the same kpoints as testcases.jl
    # and Ecut = 25
    ref_lda = [
        [-0.178566465714968, 0.261882541175914, 0.261882541178847, 0.261882541181782,
          0.354070367072414, 0.354070367076363, 0.354070367080310, 0.376871160884678],
        [-0.127794342370963, 0.064395861472044, 0.224958824747686, 0.224958824750934,
          0.321313617512188, 0.388442495007398, 0.388442495010722, 0.542078732298094],
        [-0.108449612789883, 0.077125812982728, 0.172380374761464, 0.172380374766260,
          0.283802499666810, 0.329872296009131, 0.525606867582028, 0.525606867585921],
        [-0.058089253154566, 0.012364292440522, 0.097350168867990, 0.183765652148129,
          0.314593174568090, 0.470869435132365, 0.496966579772700, 0.517009645871194],
    ]
    ref_etot = -7.911817522631488
    n_bands = length(ref_lda[1])

    fft_size = grid_size * ones(3)
    Si = Species(silicon.atnum, psp=load_psp(silicon.psp))
    model = model_dft(Array{T}(silicon.lattice), [:lda_x, :lda_c_vwn], Si => silicon.positions)
    basis = PlaneWaveBasis(model, Ecut, silicon.kcoords, silicon.ksymops; fft_size=fft_size)
    ham = Hamiltonian(basis, guess_density(basis, Si => silicon.positions))

    scfres = self_consistent_field(ham, n_bands, tol=scf_tol,
                                   eigensolver=lobpcg_hyper, n_ep_extra=n_noconv_check, diagtol=lobpcg_tol)

    for ik in 1:length(silicon.kcoords)
        @test eltype(scfres.orben[ik]) == T
        @test eltype(scfres.Psi[ik]) == Complex{T}
    end
    for ik in 1:length(silicon.kcoords)
        # Ignore last few bands, because these eigenvalues are hardest to converge
        # and typically a bit random and unstable in the LOBPCG
        @test maximum(ref_lda[ik][1:n_bands - n_ignored] - scfres.orben[ik][1:n_bands - n_ignored]) < test_tol
    end

    energies = scfres.energies
    energies[:Ewald] = energy_nuclear_ewald(model.lattice, Si => silicon.positions)
    energies[:PspCorrection] = energy_nuclear_psp_correction(model.lattice,
                                                             Si => silicon.positions)
    @test sum(values(energies)) ≈ ref_etot atol=test_tol
end


function run_silicon_pbe(T ;Ecut=5, test_tol=1e-6, n_ignored=0, grid_size=15, scf_tol=1e-6)
    # These values were computed using ABINIT with the same kpoints as testcases.jl
    # and Ecut = 25
    ref_pbe = [
        [-0.181210259413818, 0.258840553222639, 0.258840553225549, 0.258840553228459,
          0.351692348652324, 0.351692348656259, 0.351692348660193, 0.380606400669216,
          0.540705881744348, 0.540705883460555],
        [-0.130553299114991, 0.062256443775155, 0.221871391287580, 0.221871391290802,
          0.322398722411882, 0.386194327436667, 0.386194327439986, 0.546859898649217,
          0.550571701390781, 0.550571701394327],
        [-0.111170738096744, 0.074494899973125, 0.169461730083372, 0.169461730088140,
          0.284305392082236, 0.330468937070505, 0.524509288492752, 0.524509288496625,
          0.616964090764029, 0.619623658242765],
        [-0.061054203629684, 0.009700769243041, 0.095769985640881, 0.180784778430457,
          0.315000287382235, 0.471042322838057, 0.495281775946584, 0.517469860611792,
          0.530124341745161, 0.539044739392045],
    ]
    ref_etot = -7.854477356672080
    n_bands = length(ref_pbe[1])

    fft_size = grid_size * ones(3)
    Si = Species(silicon.atnum, psp=load_psp("hgh/pbe/si-q4"))
    model = model_dft(Array{T}(silicon.lattice), [:gga_x_pbe, :gga_c_pbe], Si => silicon.positions)
    basis = PlaneWaveBasis(model, Ecut, silicon.kcoords, silicon.ksymops; fft_size=fft_size)
    ham = Hamiltonian(basis, guess_density(basis, Si => silicon.positions))

    scfres = self_consistent_field(ham, n_bands, tol=scf_tol)

    for ik in 1:length(silicon.kcoords)
        @test eltype(scfres.orben[ik]) == T
        @test eltype(scfres.Psi[ik]) == Complex{T}
        println(ik, "  ", abs.(ref_pbe[ik] - scfres.orben[ik]))
    end
    for ik in 1:length(silicon.kcoords)
        # Ignore last few bands, because these eigenvalues are hardest to converge
        # and typically a bit random and unstable in the LOBPCG
        diff = abs.(ref_pbe[ik] - scfres.orben[ik])
        @test maximum(diff[1:n_bands - n_ignored]) < test_tol
    end

    energies = scfres.energies
    energies[:Ewald] = energy_nuclear_ewald(model.lattice, Si => silicon.positions)
    energies[:PspCorrection] = energy_nuclear_psp_correction(model.lattice,
                                                             Si => silicon.positions)
    @test sum(values(energies)) ≈ ref_etot atol=test_tol
end
