using DFTK
using ForwardDiff
using Test
using ComponentArrays

include("testcases.jl")


# @testset "Force derivatives using ForwardDiff" begin
#     function compute_force(ε1, ε2; metal=false)
#         T = promote_type(typeof(ε1), typeof(ε2))
#         pos = [[1.01, 1.02, 1.03] / 8, -ones(3) / 8 + ε1 * [1., 0, 0] + ε2 * [0, 1., 0]]
#         if metal
#             # Silicon reduced HF is metallic
#             model = model_DFT(Matrix{T}(silicon.lattice), silicon.atoms, pos, [];
#                               temperature=1e-3)
#         else
#             model = model_LDA(Matrix{T}(silicon.lattice), silicon.atoms, pos)
#         end
#         basis = PlaneWaveBasis(model; Ecut=5, kgrid=[2, 2, 2], kshift=[0, 0, 0])
# 
#         is_converged = DFTK.ScfConvergenceDensity(1e-10)
#         scfres = self_consistent_field(basis; is_converged,
#                                        response=ResponseOptions(verbose=true))
#         compute_forces_cart(scfres)
#     end
# 
#     F = compute_force(0.0, 0.0)
#     derivative_ε1_fd = let ε1 = 1e-5
#         (compute_force(ε1, 0.0) - F) / ε1
#     end
#     derivative_ε1 = ForwardDiff.derivative(ε1 -> compute_force(ε1, 0.0), 0.0)
#     @test norm(derivative_ε1 - derivative_ε1_fd) < 1e-4
# 
#     derivative_ε2_fd = let ε2 = 1e-5
#         (compute_force(0.0, ε2) - F) / ε2
#     end
#     derivative_ε2 = ForwardDiff.derivative(ε2 -> compute_force(0.0, ε2), 0.0)
#     @test norm(derivative_ε2 - derivative_ε2_fd) < 1e-4
# 
#     @testset "Multiple partials" begin
#         grad = ForwardDiff.gradient(v -> compute_force(v...)[1][1], [0.0, 0.0])
#         @test abs(grad[1] - derivative_ε1[1][1]) < 1e-4
#         @test abs(grad[2] - derivative_ε2[1][1]) < 1e-4
# 
#         jac = ForwardDiff.jacobian(v -> compute_force(v...)[1], [0.0, 0.0])
#         @test norm(grad - jac[1, :]) < 1e-10
#     end
# 
#     @testset "Derivative for metals" begin
#         F = compute_force(0.0, 0.0; metal=true)
#         derivative_ε1_fd = let ε1 = 1e-5
#             (compute_force(ε1, 0.0; metal=true) - F) / ε1
#         end
#         derivative_ε1 = ForwardDiff.derivative(ε1 -> compute_force(ε1, 0.0; metal=true), 0.0)
#         @test norm(derivative_ε1 - derivative_ε1_fd) < 1e-4
#     end
# end

@testset "scfres PSP sensitivity using ForwardDiff" begin
    function compute_band_energies(ε::T) where {T}
        psp  = load_psp("hgh/lda/al-q3")
        rloc = convert(T, psp.rloc)

        pspmod = PspHgh(psp.Zion, rloc,
                        psp.cloc, psp.rp .+ [0, ε], psp.h;
                        psp.identifier, psp.description)
        atoms = fill(ElementPsp(aluminium.atnum, psp=pspmod), length(aluminium.positions))
        model = model_LDA(Matrix{T}(aluminium.lattice), atoms, aluminium.positions,
                          temperature=1e-2, smearing=Smearing.Gaussian())
        basis = PlaneWaveBasis(model; Ecut=5, kgrid=[2, 2, 2], kshift=[0, 0, 0])

        is_converged = DFTK.ScfConvergenceDensity(1e-10)
        scfres = self_consistent_field(basis; is_converged,
                                       damping=0.6, response=ResponseOptions(verbose=true))

        ComponentArray(
           eigenvalues=hcat([ev[1:end-3] for ev in scfres.eigenvalues]...),
           ρ=scfres.ρ
        )
    end

    derivative_ε = let ε = 1e-4
        (compute_band_energies(ε) - compute_band_energies(-ε)) / 2ε
    end
    derivative_fd = ForwardDiff.derivative(compute_band_energies, 0.0)
    @show norm(derivative_fd - derivative_ε)
    @test norm(derivative_fd - derivative_ε) < 1e-4


    reference_eigenvalues = [-0.2384551782408563 -0.23768481751589635 -0.081074581308665 -0.08166199596881459 0.047714921091244616 0.048967308889071935; -0.15067986242626097 -0.14986768712936221 -0.08107458130611957 -0.08166199596783183 0.047714921091394524 0.04896730888929108; 0.6629479453031047 0.664506017966151 0.022921680648071294 0.022387823681221927 0.04771492109123428 0.04896730888908208; 0.6629479453028844 0.6645060179671527 0.02292168064160855 0.022387823681656108 0.04771492109152041 0.04896730888923781; 0.44050788755569475 0.4409056731316783 0.6044385260810143 0.6053961633672325 0.2159091597098529 0.21653079864743904; -0.1980020885042895 -0.19748878201036996 0.6044385260779945 0.6053961633670825 0.21590915971298566 0.21653079864768948; -0.19800208850434187 -0.19748878201033093 0.45288115660538697 0.452098594280435 0.2159091597126262 0.21653079864893665; 0.8039330250087587 0.8057041389045898 0.45288115660392825 0.45209859428132204 0.21590915971289942 0.21653079864772395; 0.8039330250157423 0.8057041389003357 0.1387111544516983 0.13963075050454654 0.4035129054141271 0.403822080212611; -0.029900905153639013 -0.026967018525908295 0.13871115445344695 0.13963075050509072 0.40351290541559764 0.40382208021203125]

    @show norm(derivative_fd.eigenvalues - reference_eigenvalues)
end
