@testmodule RunSCF begin
using Test
using DFTK
using DFTK: mpi_sum

function run_scf_and_compare(T, basis, ref_evals, ref_etot; n_ignored=0, test_tol=1e-6,
                             scf_ene_tol=1e-6, test_etot=true, kwargs...)
    n_kpt    = length(ref_evals)
    n_bands  = length(ref_evals[1])
    kpt_done = zeros(Bool, n_kpt)

    nbandsalg    = AdaptiveBands(basis.model, n_bands_converge=n_bands)
    is_converged = DFTK.ScfConvergenceEnergy(scf_ene_tol)
    scfres = self_consistent_field(basis; is_converged, nbandsalg, kwargs...)
    for (ik, ik_global) in enumerate(basis.krange_thisproc_allspin)
        @test eltype(scfres.eigenvalues[ik]) == T
        @test eltype(scfres.ψ[ik]) == Complex{T}
        # println(ik_global, "  ", abs.(ref_evals[ik] - scfres.eigenvalues[ik][1:n_bands]))
    end
    for (ik, ik_global) in enumerate(basis.krange_thisproc_allspin)
        # Ignore last few bands, because these eigenvalues are hardest to converge
        # and typically a bit random and unstable in the LOBPCG
        diff = abs.(ref_evals[ik_global] - scfres.eigenvalues[ik][1:n_bands])
        @test maximum(diff[1:n_bands - n_ignored]) < test_tol

        kpt_done[ik_global] = true
    end
    @test mpi_sum(sum(kpt_done), basis.comm_kpts) == n_kpt

    @test eltype(scfres.energies.total) == T
    test_etot && (@test scfres.energies.total ≈ ref_etot atol=test_tol)

    scfres
end
end
