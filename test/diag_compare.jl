@testitem "Comparison of diagonalisaton procedures" begin
    using DFTK

    function test_solver(reference, eigensolver, prec_type)
        @testset "$eigensolver with $prec_type" begin
            nev = length(reference.λ[1])
            res = diagonalize_all_kblocks(eigensolver, reference.ham, nev; prec_type)
            @test res.λ ≈ reference.λ
        end
    end

    lattice = Float64[5 0 0; 0 0 0; 0 0 0]
    model = Model(lattice; terms=[Kinetic()])
    basis = PlaneWaveBasis(model; Ecut=100, kgrid=(1, 1, 1))
    ham = Hamiltonian(basis)
    reference = merge(diagonalize_all_kblocks(diag_full, ham, 6), (; ham))

    test_solver(reference, diag_full,    PreconditionerTPA)
    test_solver(reference, diag_full,    PreconditionerNone)
    test_solver(reference, lobpcg_hyper, PreconditionerTPA)
end
