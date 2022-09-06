using DFTK
using Test

include("testcases.jl")

function test_addiis(testcase; temperature=0, Ecut=10,
                   kgrid=[3, 3, 3], δ=1e-4)
  model_kwargs = (; temperature)
  basis_kwargs = (; kgrid, Ecut)

  tol = 1e-6
  model = model_LDA(testcase.lattice, testcase.atoms, testcase.positions;
                    model_kwargs...)
  basis = PlaneWaveBasis(model; basis_kwargs...)
  scf_res_addiis = self_consistent_field(basis;solver = scf_anderson_solver(;δ=δ),mixing=SimpleMixing(),tol=tol)
  scf_res_rdiis = self_consistent_field(basis;solver = scf_anderson_solver(;δ=0.0,maxcond=1e-6),mixing=SimpleMixing(),tol=tol,maxiter=30) #restarted DIIS
  if scf_res_rdiis.converged
    @test isapprox(scf_res_rdiis.energies.total, scf_res_addiis.energies.total; rtol=tol)
  else
    scf_res_rdiis = self_consistent_field(basis;solver = scf_anderson_solver(;δ=0.0,maxcond=1e-6),mixing=LdosMixing(),tol=tol,maxiter=30) #restarted DIIS
    @test isapprox(scf_res_rdiis.energies.total, scf_res_addiis.energies.total; rtol=tol)
  end
end

@testset "AD-DIIS" begin
  for (case, temperatures,δs) in [(silicon, (0, 0.03), (1e-6,1e-4,1e-2,0.1)), (aluminium, (0.01, ), (1e-6,1e-4,1e-2,0.1))]
    for temperature in temperatures
      for δ in δs
        test_addiis(case; temperature, δ)
      end
    end
  end
end