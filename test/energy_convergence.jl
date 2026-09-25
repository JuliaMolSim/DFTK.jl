@testitem "Testing ScfConvergenceEnergy" begin
using DFTK

function test_energy_convergence(tol=1-6)
    conv = ScfConvergenceEnergy(tol)
    
    #converged dummy histories
    info_minus = (;history_Δρ=[1e-5,1e-6], history_Etot=[0.0, -0.5 * tol])
    info_plus  = (;history_Δρ=[1e-5,1e-6], history_Etot=[0.0,  0.5 * tol])
    
    @test conv(info_minus)
    @test conv(plus)
    
    #non-converged dummy histories 
    info_minus = (;history_Δρ=[1e-5,1e-6], history_Etot=[0.0, -2.0 * tol])
    info_plus  = (;history_Δρ=[1e-5,1e-6], history_Etot=[0.0,  2.0 * tol])
    
    @test !conv(info_minus)
    @test !conv(plus)
end

@testset "Energy convergence criterium" begin
    test_energy_convergence()
end
end
