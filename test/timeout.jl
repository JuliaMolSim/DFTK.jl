@testitem "Timeout of SCF" setup=[TestCases] begin
    using DFTK
    using Dates
    silicon = TestCases.silicon

    model = model_LDA(silicon.lattice, silicon.atoms, silicon.positions)
    basis = PlaneWaveBasis(model; Ecut=7, kgrid=(1, 1, 1))
    is_converged = DFTK.ScfConvergenceDensity(1e-11)

    function sleep_callback(info)
        sleep(1)
        return info
    end
    callback = ScfDefaultCallback() ∘ sleep_callback

    @test_warn "SCF not converged." begin
        maxtime = Dates.Millisecond(10)
        scfres = self_consistent_field(basis; is_converged, callback, maxtime)
        @test scfres.timeout
    end    
end
