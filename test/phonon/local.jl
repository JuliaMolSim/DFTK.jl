# TODO: Will need to be factorized with `helpers.jl` functions.
# Needs a tiny bit of work to compute only necessary quantities, e.g. no need for SCF for
# Ewald.
@testsetup module PhononLocal
using DFTK
using Test
using LinearAlgebra
using ForwardDiff
using Random
using ..Phonon

# No exchange-correlation and only a local potential.
function model_tested(lattice::AbstractMatrix,
                      atoms::Vector{<:DFTK.Element},
                      positions::Vector{<:AbstractVector};
                      extra_terms=[], kinetic_blowup=BlowupIdentity(), kwargs...)
    @assert !(:terms in keys(kwargs))
    terms = [Kinetic(; blowup=kinetic_blowup),
             AtomicLocal(),
             Ewald(),
             PspCorrection(),
             Hartree(),
             extra_terms...]
    if :temperature in keys(kwargs) && kwargs[:temperature] != 0
        terms = [terms..., Entropy()]
    end
    Model(lattice, atoms, positions; model_name="atomic", terms, kwargs...)
end


function are_approx_frequencies(ω_uc, ω_ref; tol=1e-10)
    # Because three eigenvalues should be close to zero and the square root near
    # zero decrease machine accuracy, we expect at least ``3×2×2 - 3 = 9``
    # eigenvalues to have norm related to the accuracy of the SCF convergence
    # parameter and the rest to be larger.
    n_dim = 3
    n_atoms = length(ω_uc) ÷ 3

    @test count(abs.(ω_uc - ω_ref) .< sqrt(tol)) ≥ n_dim*n_atoms - n_dim
    @test count(sqrt(tol) .< abs.(ω_uc - ω_ref) .< tol) ≤ n_dim
end

function test_frequencies(testcase; ω_ref=nothing)
    Ecut  = 7
    kgrid = [2, 1, 3]
    qpoints = Phonon.generate_supercell_qpoints(; supercell_size=kgrid).qpoints
    scf_tol = 1e-12
    χ0_tol  = scf_tol/10
    is_converged = DFTK.ScfConvergenceDensity(scf_tol)
    determine_diagtol = DFTK.ScfDiagtol(diagtol_max=scf_tol)
    scf_kwargs = (; is_converged, determine_diagtol)

    model = model_tested(testcase.lattice, testcase.atoms, testcase.positions;
                         symmetries=false, testcase.temperature)
    nbandsalg = AdaptiveBands(model; occupation_threshold=1e-10)
    scf_kwargs = merge(scf_kwargs, (; nbandsalg))
    basis = PlaneWaveBasis(model; Ecut, kgrid)
    scfres = self_consistent_field(basis; scf_kwargs...)
    ω_uc = sort!(reduce(vcat, map(qpoints) do q
        dynamical_matrix = compute_dynmat(scfres; q, tol=χ0_tol)
        phonon_frequencies(basis, dynamical_matrix)
    end))

    if !isnothing(ω_ref)
        return are_approx_frequencies(ω_uc, ω_ref; tol=10*scf_tol)
    end

    supercell_size = kgrid
    kgrid_supercell = [1, 1, 1]

    supercell = create_supercell(testcase.lattice, testcase.atoms, testcase.positions,
                                 supercell_size)
    model_supercell = model_tested(supercell.lattice, supercell.atoms, supercell.positions;
                                   symmetries=false, testcase.temperature)
    nbandsalg = AdaptiveBands(model_supercell; occupation_threshold=1e-10)
    scf_kwargs = merge(scf_kwargs, (; nbandsalg))
    basis_supercell = PlaneWaveBasis(model_supercell; Ecut, kgrid=kgrid_supercell)

    scfres = self_consistent_field(basis_supercell; scf_kwargs...)
    dynamical_matrix = compute_dynmat(scfres; tol=χ0_tol)
    ω_sc = sort(phonon_frequencies(basis_supercell, dynamical_matrix))

    return are_approx_frequencies(ω_uc, ω_sc; tol=10*scf_tol)
end
end

@testitem "Phonon: Local term: comparison to ref testcase" #=
    =#    tags=[:phonon, :dont_test_mpi] setup=[Phonon, PhononLocal, TestCases] begin
    ω_ref = [ -0.0006800428228943786
               0.00025946473743646887
               0.000327857002617677
              97.24345593504349
             102.60823069748295
             102.60823069749723
             114.27333066498551
             114.27333066500222
             186.1463261487283
             199.28011875220716
             199.2801187522361
             222.13926530548198
             222.13926530552334
             294.27830868298577
             302.42610677479854
             302.42610677482594
             307.7446637388342
             307.74466373885156]
    PhononLocal.test_frequencies(TestCases.aluminium_primitive; ω_ref)
end

@testitem "Phonon: Local term: comparison to supercell" #=
    =#    tags=[:phonon, :dont_test_mpi, :slow] setup=[Phonon, PhononLocal, TestCases] begin
    PhononLocal.test_frequencies(TestCases.aluminium_primitive)
end
