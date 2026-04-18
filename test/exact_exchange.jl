@testitem "Comparison of VanillaExx to AceExx" tags=[:exx,:dont_test_mpi,:minimal] setup=[TestCases] begin
    using DFTK
    using LinearAlgebra
    using .TestCases: silicon

    function test_acexx_consistency(; kgrid=[1, 2, 3], kshift=[0, 1, 0]/2, Ecut=10,
                                      n_empty=3, atol=1e-8, spin_polarization=:none,
                                      kernel=Coulomb(ProbeCharge()))
        Si = ElementPsp(14, load_psp(silicon.psp_upf))
        model = Model(silicon.lattice, [Si, Si], silicon.positions;
                      spin_polarization, symmetries=true, terms=[ExactExchange(; kernel)])
        basis = PlaneWaveBasis(model; Ecut, kgrid=MonkhorstPack(kgrid; kshift))

        n_bands = div(silicon.n_electrons, 2, RoundUp)
        filled_occ = DFTK.filled_occupation(model)
        ψ = [Matrix(qr(randn(ComplexF64, length(G_vectors(basis, kpt)), n_bands + n_empty)).Q)
             for kpt in basis.kpoints]
        occupation  = [filled_occ * append!(rand(n_bands), zeros(n_empty))
                       for _ = 1:length(basis.kpoints)]
        occ_scaling = length(basis.kpoints) * silicon.n_electrons / sum(sum(occupation))
        occupation  = [occ * occ_scaling for occ in occupation]

        (; energies ) = energy_hamiltonian(basis, ψ, occupation; exxalg=VanillaExx())
        energies2     = DFTK.energy(basis, ψ, occupation; exxalg=VanillaExx()).energies
        energies_ace  = DFTK.energy(basis, ψ, occupation; exxalg=AceExx()).energies

        @test abs(energies.total - energies2.total)    < atol
        @test abs(energies.total - energies_ace.total) < atol
    end

    for kernel in (Coulomb(ProbeCharge()), ShortRangeCoulomb(), SphericallyTruncatedCoulomb())
        test_acexx_consistency(; kgrid=(1, 1, 1), kshift=(0, 0, 0))
        test_acexx_consistency(; kgrid=(1, 1, 1), kshift=(0, 0, 0), spin_polarization=:collinear)
    end
end
