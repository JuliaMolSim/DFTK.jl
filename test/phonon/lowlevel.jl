@testitem "Phonon: Shifting functions" tags=[:phonon, :dont_test_mpi] begin
    using DFTK
    using DFTK: k_to_kpq_permutation
    using LinearAlgebra

    tol = 1e-12

    positions = [[0.0, 0.0, 0.0]]
    n_scell = 2
    for i = 1:n_scell-1
        push!(positions, i * ones(3) / n_scell)
    end
    n_atoms = length(positions)

    lattice = 5 * n_atoms * rand(3, 3)

    X = ElementGaussian(1.0, 0.5, :X)
    atoms = [X for _ in positions]

    model = Model(lattice, atoms, positions; n_electrons=n_atoms, symmetries=false,
                  spin_polarization=:collinear)
    kgrid = rand(2:10, 3)
    k1, k2, k3 = kgrid
    basis = PlaneWaveBasis(model; Ecut=100, kgrid)

    # Random `q` shift
    q0 = rand(basis.kpoints).coordinate
    ishift = [rand(-k1*2:k1*2), rand(-k2*2:k2*2), rand(-k3*2:k3*2)]
    q = Vec3(q0 .* ishift)

    @testset "Ordering function" begin
        k_to_k_plus_q = k_to_kpq_permutation(basis, q)
        kcoords = getfield.(basis.kpoints, :coordinate)
        for (ik, kcoord) in enumerate(kcoords)
            @test mod.(kcoord + q .- tol, 1) ≈ mod.(kcoords[k_to_k_plus_q[ik]] .- tol, 1)
        end
    end
end
