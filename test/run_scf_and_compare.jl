using Test
using DFTK

function run_scf_and_compare(T, basis, ref_evals, ref_etot;
                             n_ignored=0, test_tol=1e-6, scf_tol=1e-6, test_etot=true,
                             n_kpt_rounds=2, kwargs...)
    n_kpt = length(ref_evals)
    n_bands = length(ref_evals[1])
    kpt_done = zeros(Bool, n_kpt * n_kpt_rounds)

    scfres = self_consistent_field(basis; tol=scf_tol, n_bands=n_bands, kwargs...)
    for ik in 1:basis.krange_thisproc
        @test eltype(scfres.eigenvalues[ik]) == T
        @test eltype(scfres.ψ[ik]) == Complex{T}
        # println(ik, "  ", abs.(ref_evals[mod1(ik, n_kpt)] - scfres.eigenvalues[ik][1:n_bands]))
    end
    for ik in 1:basis.krange_thisproc
        # Ignore last few bands, because these eigenvalues are hardest to converge
        # and typically a bit random and unstable in the LOBPCG
        diff = abs.(ref_evals[mod1(ik, n_kpt)] - scfres.eigenvalues[ik][1:n_bands])
        @test maximum(diff[1:n_bands - n_ignored]) < test_tol

        kpt_done[ik] = true
    end
    @test mpi_sum!(sum(kpt_done), basis.comm_k) == n_kpt * n_kpt_rounds

    @test eltype(scfres.energies.total) == T
    test_etot && (@test scfres.energies.total ≈ ref_etot atol=test_tol)

    scfres
end
