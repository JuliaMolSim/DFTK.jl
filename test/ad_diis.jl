using DFTK
using Test

include("testcases.jl")

function test_addiis(testcase; temperature=0, Ecut=10,
                   kgrid=[3, 3, 3])
  model_kwargs = (; temperature)
  basis_kwargs = (; kgrid, Ecut)

  tol = 1e-6
  model = model_LDA(testcase.lattice, testcase.atoms, testcase.positions;
                    model_kwargs...)
  basis = PlaneWaveBasis(model; basis_kwargs...)
  scf_res_addiis = self_consistent_field(basis;solver = scf_anderson_solver(),mixing=SimpleMixing(),tol=tol)
  scf_res_rdiis = self_consistent_field(basis;solver = scf_anderson_solver(;δ=0.0,maxcond=1e-6),mixing=SimpleMixing(),tol=tol,maxiter=100) #restarted DIIS
  if scf_res_rdiis.converged
    @test abs(scf_res_rdiis.energies.total - scf_res_addiis.energies.total) < tol
  else
    scf_res_rdiis = self_consistent_field(basis;solver = scf_anderson_solver(;δ=0.0,maxcond=1e-6),mixing=LdosMixing(),tol=tol,maxiter=100) #restarted DIIS
    @test abs(scf_res_rdiis.energies.total - scf_res_addiis.energies.total) < tol
  end
end

@testset "AD-DIIS" begin
  for (case, temperatures) in [(silicon, (0, 0.03)), (aluminium, (0.01, ))]
    for temperature in temperatures
      test_addiis(case; temperature)
    end
  end
end