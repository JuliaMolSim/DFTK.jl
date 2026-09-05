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
        test_acexx_consistency(; kgrid=(1, 1, 1), kshift=(0, 0, 0), kernel)
        test_acexx_consistency(; kgrid=(1, 1, 1), kshift=(0, 0, 0), kernel, spin_polarization=:collinear)
        test_acexx_consistency(; kgrid=(2, 1, 1), kshift=(0, 0, 0), kernel)
        test_acexx_consistency(; kgrid=(2, 1, 2), kshift=(1/2, 1/2, 1/2), kernel)
    end
end

@testitem "Exact exchange with explicit k-point grids" tags=[:exx, :dont_test_mpi, :minimal] setup=[TestCases] begin
    using DFTK
    using LinearAlgebra
    using .TestCases: silicon

    # The same half-shifted 2x2x2 grid once as MonkhorstPack and once as ExplicitKpoints
    # with coordinates outside [-1/2, 1/2). The exchange energy of the same orbitals has
    # to agree, which requires k' = k - q to be found modulo reciprocal lattice vectors.
    N  = [2, 2, 2]
    Si = ElementPsp(silicon.atnum, load_psp(silicon.psp_upf))
    kcoords_explicit = vec([([1, 1, 1]/2 .+ [i, j, k]) ./ N
                            for i = 0:N[1]-1, j = 0:N[2]-1, k = 0:N[3]-1])
    @test any(k -> maximum(k) ≥ 0.5, kcoords_explicit)

    # Orbitals defined through their Cartesian G+k vectors (Gaussians around some centers),
    # such that they do not depend on the representative chosen for equivalent k-points.
    centers = [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.5]]
    function exchange_energy(kgrid)
        model = Model(silicon.lattice, [Si, Si], silicon.positions;
                      terms=[ExactExchange(; kernel=Coulomb(ProbeCharge()))])
        basis = PlaneWaveBasis(model; Ecut=5, kgrid)
        ψ = map(basis.kpoints) do kpt
            Gpk = Gplusk_vectors_cart(basis, kpt)
            coefficients = stack(exp.(-norm.(Gpk .- Ref(c)) .^ 2) for c in centers)
            Matrix(qr(complex(coefficients)).Q)
        end
        occupation = [[2.0, 2.0, 1.5, 0.5] for _ in basis.kpoints]
        DFTK.energy(basis, ψ, occupation; exxalg=VanillaExx()).energies.total
    end
    E_mp       = exchange_energy(MonkhorstPack(N; kshift=[1, 1, 1]/2))
    E_explicit = exchange_energy(ExplicitKpoints(kcoords_explicit))
    @test E_mp ≈ E_explicit rtol=1e-10

    # A k-point set which is not closed under k - k' cannot be used for exact exchange
    model = Model(silicon.lattice, [Si, Si], silicon.positions; terms=[ExactExchange()])
    @test_throws ErrorException PlaneWaveBasis(model; Ecut=5,
                                               kgrid=ExplicitKpoints([[0, 0, 0], [1/4, 0, 0]]))
end

@testitem "Exact exchange energy of a k-point grid against its supercell" #=
        =# tags=[:exx, :dont_test_mpi] setup=[TestCases] begin
    using DFTK
    using FastGaussQuadrature
    using LinearAlgebra
    using .TestCases: silicon

    # For fixed orbitals the exchange energy per unit cell of a k-point grid has to agree
    # with the energy of the same orbitals in the supercell at Γ. This holds for all
    # kernels defined through the supercell corresponding to the k-point grid.
    Si = ElementPsp(silicon.atnum, load_psp(silicon.psp_upf))
    kgrid   = [2, 1, 2]
    n_bands = 4
    for kernel in (Coulomb(ProbeCharge()), Coulomb(VoxelAveraged()),
                   SphericallyTruncatedCoulomb(), WignerSeitzTruncatedCoulomb())
        model = Model(silicon.lattice, [Si, Si], silicon.positions;
                      terms=[ExactExchange(; kernel)])
        basis = PlaneWaveBasis(model; Ecut=5, kgrid)
        basis_supercell = cell_to_supercell(basis)

        ψ = [Matrix(qr(randn(ComplexF64, length(G_vectors(basis, kpt)), n_bands)).Q)
             for kpt in basis.kpoints]
        occupation = [[2.0, 2.0, 1.5, 0.5] for _ in basis.kpoints]
        ψ_supercell   = [cell_to_supercell(ψ, basis, basis_supercell)]
        occ_supercell = [reduce(vcat, occupation)]

        E = DFTK.energy(basis, ψ, occupation; exxalg=VanillaExx()).energies.total
        E_supercell = DFTK.energy(basis_supercell, ψ_supercell, occ_supercell;
                                  exxalg=VanillaExx()).energies.total
        @test prod(kgrid) * E ≈ E_supercell rtol=1e-9
    end
end
