using DFTK
using Test

include("testcases.jl")


@testset "Energy is exact for supersampling>2 without XC" begin
    for XC in (true, false)
        println(XC)
        res = []
        for supersampling in (1, 2, 3)
            Ecut=3
            grid_size=15
            scf_tol=1e-10
            n_bands = 10
            kcoords = [[.2, .3, .4]]
            ksymops = nothing

            fft_size = determine_grid_size(silicon.lattice, Ecut, supersampling=supersampling, ensure_smallprimes=false)
            Si = Species(silicon.atnum, psp=load_psp(silicon.psp))
            if XC
                model = model_dft(silicon.lattice, [:lda_x, :lda_c_vwn], Si => silicon.positions)
            else
                model = model_reduced_hf(silicon.lattice, Si => silicon.positions)
            end

            basis = PlaneWaveModel(model, fft_size, Ecut, kcoords, ksymops)
            ham = Hamiltonian(basis, guess_gaussian_sad(basis, Si => silicon.positions))
            scfres = self_consistent_field!(ham, n_bands, tol=scf_tol)

            push!(res, scfres)
        end

        dictdiff(D1, D2) = norm(values(D1.energies) .- values(D2.energies))
        @test dictdiff(res[1], res[2]) > 1e-10
        if XC
            @test dictdiff(res[2], res[3]) > 1e-10
        else
            @test dictdiff(res[2], res[3]) < 1e-10
        end
    end
end
