@testitem "Energy cutoff smearing on silicon LDA" #=
    =#    tags=[:dont_test_mpi] setup=[TestCases] begin
    using DFTK
    using LinearAlgebra
    silicon = TestCases.silicon

    # For a low Ecut, the first silicon band displays a discontinuity between the
    # X and U points. This code checks the presence of the discontinuity for
    # the standard kinetic term and checks that the same band computed with a modified
    # kinetic terms has C^2 regularity.

    model = model_LDA(silicon.lattice, silicon.atoms, silicon.positions)
    basis = PlaneWaveBasis(model, 5, silicon.kcoords, silicon.kweights)
    scfres = self_consistent_field(basis; callback=identity)

    # Kpath around one discontinuity of the first band of silicon (between X and U points)
    k_start = [0.5274, 0.0548, 0.5274]
    k_end   = [0.5287, 0.0573, 0.5287]
    kcoords = map(x->(1-x)*k_start .+ x*k_end, range(0, 1; length=100))
    δk = norm(kcoords[2] .- kcoords[1], 1)

    # Test irregularity of the standard band through its second finite diff derivative
    basis_std = PlaneWaveBasis(model, 5, silicon.kcoords, silicon.kweights; basis.fft_size)
    λ_std = vcat(compute_bands(basis_std, kcoords, n_bands=1; scfres.ρ).λ...)
    ∂2λ_std = [(λ_std[i+1] - 2*λ_std[i] + λ_std[i-1])/δk^2 for i = 2:length(kcoords)-1]

    # Compute band for given blow-up and test regularity
    function test_blowup(kinetic_blowup)
        model = model_LDA(silicon.lattice, silicon.atoms, silicon.positions; kinetic_blowup)
        basis_mod = PlaneWaveBasis(model, 5, silicon.kcoords, silicon.kweights; basis.fft_size)

        λ_mod = vcat(compute_bands(basis_mod, kcoords, n_bands=1, ρ=scfres.ρ).λ...)
        ∂2λ_mod = [(λ_mod[i+1] - 2*λ_mod[i] + λ_mod[i-1])/δk^2 for i = 2:length(kcoords)-1]
        @test norm(∂2λ_std) / norm(∂2λ_mod) > 1e4
        nothing
    end
    for blowup in (BlowupCHV(), BlowupAbinit())
        @testset "Testing $(typeof(blowup))" begin
            test_blowup(blowup)
        end
    end
end
