using DFTK
using Test
using LinearAlgebra: norm

include("testcases.jl")

@testset "Computing χ0" begin
    Ecut=100
    fft_size = [60, 1, 1]
    tol=1e-10
    ε = 1e-8
    kgrid = [1, 1, 1]
    testcase = silicon
    n_bands = 10

    spec = ElementPsp(testcase.atnum, psp=load_psp(testcase.psp))
    for temperature in (0, 0.03)
        model = model_LDA(testcase.lattice, [spec => testcase.positions], temperature=temperature)

        basis = PlaneWaveBasis(model, Ecut; kgrid=kgrid, fft_size=fft_size)

        ρ0 = guess_density(basis)
        energies, ham = energy_hamiltonian(basis, nothing, nothing;
                                           ρ=ρ0)
        V = DFTK.total_local_potential(ham)
        ρV = DFTK.next_density(ham, tol=tol, eigensolver=diag_full, n_bands=n_bands).ρ
        χ0 = compute_χ0(ham)

        # now we go change the local potential of the hamiltonian
        # TODO this is a bit of a hack...
        dV = randn(eltype(V), size(V))
        dV = dV .- mean(dV)
        DFTK.set_total_local_potential!(ham, V + ε.*dV)
        ρVpdV = DFTK.next_density(ham, tol=tol, eigensolver=diag_full, n_bands=n_bands).ρ
        diff = (ρVpdV.real - ρV.real)/ε

        predicted_diff = real(reshape(χ0*vec(dV), basis.fft_size))
        @test norm(diff - predicted_diff) < sqrt(ε)
    end
end
