@testitem "Test NonlocalProjectors" tags=[:psp] setup=[mPspUpf] begin
    using DFTK
    using LinearAlgebra

    for (element, psp) in mPspUpf.upf_pseudos
        lattice = 5 * I(3)
        el = ElementPsp(element, psp)
        atoms = [el, el, el]
        positions = [zeros(3), 1/3 .* ones(3), 2/3 .* ones(3)]
        model = model_DFT(lattice, atoms, positions; functionals=LDA())
        basis = PlaneWaveBasis(model; Ecut=5, kgrid=[2, 2, 2])

        n_bands = 7
        ψ = [DFTK.random_orbitals(basis, kpt, n_bands) for kpt in basis.kpoints]
        occ = [2.0 * ones(n_bands) for _ in basis.kpoints]
        ρ = DFTK.compute_density(basis, ψ, occ)

        energies, ham = DFTK.energy_hamiltonian(basis, ψ, occ; ρ)

        for (hamblock, ψk) in zip(ham.blocks, ψ)
            nonloc = hamblock.nonlocal_op
            Hψk = zero(ψk)
            DFTK.apply!((; fourier=Hψk), nonloc, (; fourier=ψk))

            nonloc_dense = Matrix(nonloc)
            Hψk_dense = nonloc_dense * ψk

            Pψk = nonloc.P' * ψk
            DPψk = nonloc.D * Pψk
            @assert nonloc.P * DPψk ≈ Matrix(nonloc.P) * DPψk atol=1e-10
            @assert Hψk ≈ Hψk_dense atol=1e-10
        end
    end
end
