using DFTK
using Test
using LinearAlgebra: norm

include("testcases.jl")

# @testset "Computing χ0" begin
    Ecut=3
    fft_size = [10, 10, 1]
    tol=1e-10
    ε = 1e-8
    kgrid = [3, 1, 1]
    testcase = silicon
    n_bands = 12
    using Random
    Random.seed!(0)

    spec = ElementPsp(testcase.atnum, psp=load_psp(testcase.psp))
    # for temperature in (0, 0.03)
temperature = .03
        model = model_LDA(testcase.lattice, [spec => testcase.positions], temperature=temperature)
        basis = PlaneWaveBasis(model, Ecut; kgrid=kgrid, fft_size=fft_size, enable_bzmesh_symmetry=false)

        ρ0 = guess_density(basis)
        energies, ham0 = energy_hamiltonian(basis, nothing, nothing; ρ=ρ0)
        ρ1 = DFTK.next_density(ham0, tol=tol, eigensolver=diag_full, n_bands=n_bands).ρ
        χ0 = compute_χ0(ham0)

        # Now we make the same model, but add an artificial external potential ε * dV
        dV = randn(eltype(basis), basis.fft_size)
        term_builder = basis -> DFTK.TermExternal(basis, ε .* dV)
        model = model_LDA(testcase.lattice, [spec => testcase.positions], temperature=temperature, extra_terms=[term_builder])
        basis = PlaneWaveBasis(model, Ecut; kgrid=kgrid, fft_size=fft_size, enable_bzmesh_symmetry=false)
        energies, ham = energy_hamiltonian(basis, nothing, nothing; ρ=ρ0)
        ρ2 = DFTK.next_density(ham, tol=tol, eigensolver=diag_full, n_bands=n_bands).ρ
        diff = (ρ2.real - ρ1.real)/ε

        predicted_diff = real(reshape(χ0*vec(dV), basis.fft_size))
        # @test norm(diff - predicted_diff) < sqrt(ε)

        if temperature > 0
            # Test the diagonal_only option
            χ0_diag = compute_χ0(ham0; diagonal_only=true)
            diff_diag_1 = real(reshape(χ0_diag*vec(dV), basis.fft_size))

            EVs = [eigen(Hermitian(Array(Hk))) for Hk in ham0.blocks]
            Es = [EV.values for EV in EVs]
            Vs = [EV.vectors for EV in EVs]
            occ, εF = find_occupation(basis, Es)
            diff_diag_2 = apply_chi0(ham0, dV, Vs, occ, εF, Es; diagonal_only=true)
            @test norm(diff_diag_1 - diff_diag_2) < sqrt(ε)
        end
    # end
# end
