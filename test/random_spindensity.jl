@testitem "Spin-broken silicon setup relaxes to spin-paired ground state" #=
    =#    setup=[TestCases] begin
    using DFTK
    silicon = TestCases.silicon

    function run_silicon(spin_polarization)
        model = model_PBE(silicon.lattice, silicon.atoms, silicon.positions;
                          spin_polarization, temperature=0.01)
        basis = PlaneWaveBasis(model; Ecut=7, kgrid=[2, 2, 2], kshift=[1, 1, 1] / 2)

        ρtot = total_density(guess_density(basis))
        if spin_polarization == :collinear
            spin_factors = 0.9 .+ 0.1rand(basis.fft_size...)
            ρspin = spin_factors .* ρtot
        else
            ρspin = nothing
        end
        ρ = ρ_from_total_and_spin(ρtot, ρspin)
        self_consistent_field(basis; is_converged=DFTK.ScfConvergenceEnergy(5e-6),
                              ρ, nbandsalg=FixedBands(; n_bands_converge=10));
    end

    scfres        = run_silicon(:none)
    scfres_broken = run_silicon(:collinear)
    εbroken       = scfres_broken.eigenvalues

    @test scfres.energies.total ≈ scfres_broken.energies.total atol=1e-5
    absmax(x) = maximum(abs, x)
    for (ik, kpt) in enumerate(scfres.basis.kpoints)
        kequiv = findall(kbr -> kbr.coordinate == kpt.coordinate, scfres_broken.basis.kpoints)

        for ikb in kequiv
            @test scfres.eigenvalues[ik][1:10] ≈ εbroken[ikb][1:10] atol=5e-4 norm=absmax
        end
    end
end
