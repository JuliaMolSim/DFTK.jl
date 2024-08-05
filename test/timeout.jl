@testitem "Timeout of SCF" setup=[TestCases] begin
    using DFTK
    using Dates
    using Logging
    silicon = TestCases.silicon

    model = model_LDA(silicon.lattice, silicon.atoms, silicon.positions)
    basis = PlaneWaveBasis(model; Ecut=7, kgrid=(1, 1, 1))
    is_converged = DFTK.ScfConvergenceDensity(1e-11)

    function sleep_callback(info)
        sleep(1)
        return info
    end
    callback = ScfDefaultCallback() ∘ sleep_callback

    scfres = with_logger(NullLogger()) do
        maxtime = Dates.Millisecond(10)
        self_consistent_field(basis; is_converged, callback, maxtime)
    end
    @test scfres.timedout
    @test !scfres.converged

    scfres = with_logger(NullLogger()) do
        self_consistent_field(basis; is_converged, callback, maxiter=2)
    end
    @test !scfres.timedout
    @test !scfres.converged
    @test scfres.n_iter == 2
end
