using Test
using DFTK

include("testcases.jl")

# Silicon redHF (without xc) is a metal, so we add a bit of temperature to it

# TODO There is a lot of code duplication in this file ... once we have the ABINIT reference
#      stuff in place, this should be refactored.

function run_silicon_redHF(T; Ecut=5, test_tol=1e-6, n_ignored=0, grid_size=15, scf_tol=1e-6)
    # T + Vloc + Vnloc + Vhartree
    # These values were computed using ABINIT with the same kpoints as testcases.jl
    # and Ecut = 25
    ref_redHF = [
        [0.17899118507651615, 0.6327279881297371, 0.6327279881326648, 0.6327279881356039,
         0.706557757783828, 0.7065577577877139, 0.7065577577915956, 0.7397951816714727,
         0.8532089291297222, 0.8978914445971602],
        [0.23220003663858457, 0.42189409862896016, 0.5921574659414509, 0.5921574659446628,
         0.672858189872362, 0.7372271903827399, 0.7372271903861028, 0.8643640848936627,
         0.9011792204214196, 0.9011792204356576],
        [0.2517502116803524, 0.445206025448218, 0.5328870916963034, 0.532887091701182,
         0.6211365856991057, 0.661989858948651, 0.8863951918546257, 0.8863951918584175,
         0.973261179805555, 0.9771287508158364],
        [0.30685586314464863, 0.376375429632464, 0.4438764716222098, 0.5459065154292047,
         0.651122698647485, 0.8164293660861612, 0.8515978828421051, 0.8735213568005982,
         0.8807275612483988, 0.8886454931307763]
    ]
    ref_etot = -5.440593269861395

    n_bands = length(ref_redHF[1])
    fft_size = grid_size * ones(3)
    Si = ElementPsp(silicon.atnum, psp=load_psp(silicon.psp))
    model = model_DFT(Array{T}(silicon.lattice), [Si => silicon.positions], [];
                      temperature=0.05)
    basis = PlaneWaveBasis(model, Ecut, silicon.kcoords, silicon.ksymops; fft_size=fft_size)

    scfres = self_consistent_field(basis, tol=scf_tol, n_bands=n_bands)

    for ik in 1:length(silicon.kcoords)
        @test eltype(scfres.eigenvalues[ik]) == T
        @test eltype(scfres.ψ[ik]) == Complex{T}
        println(ik, "  ", abs.(ref_redHF[ik] - scfres.eigenvalues[ik][1:n_bands]))
    end
    for ik in 1:length(silicon.kcoords)
        # Ignore last few bands, because these eigenvalues are hardest to converge
        # and typically a bit random and unstable in the LOBPCG
        diff = abs.(ref_redHF[ik] - scfres.eigenvalues[ik][1:n_bands])
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
    Si = ElementPsp(silicon.atnum, psp=load_psp(silicon.psp))
    model = model_DFT(Array{T}(silicon.lattice), [Si => silicon.positions], [:lda_x, :lda_c_vwn])
    basis = PlaneWaveBasis(model, Ecut, silicon.kcoords, silicon.ksymops; fft_size=fft_size)

    scfres = self_consistent_field(basis; n_bands=n_bands, tol=scf_tol,
                                   eigensolver=lobpcg_hyper, n_ep_extra=n_noconv_check, diagtol=lobpcg_tol)

    for ik in 1:length(silicon.kcoords)
        @test eltype(scfres.eigenvalues[ik]) == T
        @test eltype(scfres.ψ[ik]) == Complex{T}
    end
    for ik in 1:length(silicon.kcoords)
        # Ignore last few bands, because these eigenvalues are hardest to converge
        # and typically a bit random and unstable in the LOBPCG
        @test maximum(ref_lda[ik][1:n_bands - n_ignored] - scfres.eigenvalues[ik][1:n_bands - n_ignored]) < test_tol
    end

    energies = scfres.energies
    @test sum(values(energies)) ≈ ref_etot atol=test_tol
    @test eltype(sum(values(scfres.energies))) == T
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
    Si = ElementPsp(silicon.atnum, psp=load_psp("hgh/pbe/si-q4"))
    model = model_DFT(Array{T}(silicon.lattice), [Si => silicon.positions], [:gga_x_pbe, :gga_c_pbe])
    basis = PlaneWaveBasis(model, Ecut, silicon.kcoords, silicon.ksymops; fft_size=fft_size)

    scfres = self_consistent_field(basis; tol=scf_tol, n_bands=n_bands)

    for ik in 1:length(silicon.kcoords)
        @test eltype(scfres.eigenvalues[ik]) == T
        @test eltype(scfres.ψ[ik]) == Complex{T}
        println(ik, "  ", abs.(ref_pbe[ik] - scfres.eigenvalues[ik][1:n_bands]))
    end
    for ik in 1:length(silicon.kcoords)
        # Ignore last few bands, because these eigenvalues are hardest to converge
        # and typically a bit random and unstable in the LOBPCG
        diff = abs.(ref_pbe[ik] - scfres.eigenvalues[ik][1:n_bands])
        @test maximum(diff[1:n_bands - n_ignored]) < test_tol
    end

    energies = scfres.energies
    @test sum(values(energies)) ≈ ref_etot atol=test_tol
    @test eltype(sum(values(scfres.energies))) == T
end
