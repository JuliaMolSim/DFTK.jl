using Test
using DFTK

@testset "Spin-broken silicon setup relaxes to spin-paired ground state" begin
    function run_silicon(spin_polarization)
        Si = ElementPsp(silicon.atnum, psp=load_psp(silicon.psp))
        model = model_LDA(silicon.lattice, [Si => silicon.positions],
                          spin_polarization=spin_polarization, temperature=0.01)

        Ecut = 7
        basis = PlaneWaveBasis(model, Ecut; kgrid=[2, 2, 2])

        if spin_polarization == :collinear
            ρspin = from_real(basis, 2.0rand(basis.fft_size...))
        else
            ρspin = nothing
        end
        self_consistent_field(basis, tol=1e-5, ρspin=ρspin, n_bands=10);
    end

    scfres = run_silicon(:none)
    scfres_broken = run_silicon(:collinear)
    εbroken = scfres_broken.eigenvalues

    n_kpt = length(scfres.basis.kpoints)
    @test scfres.energies.total ≈ scfres_broken.energies.total atol=1e-5
    for ik in 1:n_kpt
        @test scfres.eigenvalues[ik][1:10] ≈ εbroken[ik][1:10]         atol=1e-4
        @test scfres.eigenvalues[ik][1:10] ≈ εbroken[ik + n_kpt][1:10] atol=1e-4
    end
end
