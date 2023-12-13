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
    determine_diagtol = DFTK.ScfDiagtol(diagtol_max=scf_tol)
    scf_kwargs = (; tol=scf_tol, determine_diagtol)

    model = model_tested(testcase.lattice, testcase.atoms, testcase.positions;
                         symmetries=false, testcase.temperature)
    nbandsalg = AdaptiveBands(model; occupation_threshold=1e-10)
    scf_kwargs = merge(scf_kwargs, (; nbandsalg))
    basis = PlaneWaveBasis(model; Ecut, kgrid)
    scfres = self_consistent_field(basis; scf_kwargs...)
    ω_uc = sort!(reduce(vcat, map(qpoints) do q
        dynamical_matrix = compute_dynmat(scfres; q, tol=χ0_tol)
        phonon_modes_cart(basis, dynamical_matrix).frequencies
    end))

    if !isnothing(ω_ref)
        return are_approx_frequencies(ω_uc, ω_ref; tol=10scf_tol)
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
    ω_sc = sort(phonon_modes_cart(basis_supercell, dynamical_matrix).frequencies)

    return are_approx_frequencies(ω_uc, ω_sc; tol=10scf_tol)
end
end

@testitem "Phonon: Local term: comparison to ref testcase" #=
    =#    tags=[:phonon, :dont_test_mpi] setup=[Phonon, PhononLocal, TestCases] begin
    # Values computed offline with automatic differentiation.
    ω_ref = [ -3.6569888415715e-9
              -3.6569888415715e-9
              -2.263180017613055e-9
               0.000443073786433812
               0.0004675174987222679
               0.00046751749874345965
               0.000520667604960504
               0.0005206676049755671
               0.0008481450680251938
               0.0009079870302639688
               0.0009079870302721681
               0.0010121409655813906
               0.0010121409655813906
               0.0013408306319911576
               0.0013779547317006979
               0.001377954731723582
               0.0014021878602703752
               0.001402187860292344 ]
    PhononLocal.test_frequencies(TestCases.aluminium_primitive; ω_ref)
end

@testitem "Phonon: Local term: comparison to supercell" #=
    =#    tags=[:phonon, :dont_test_mpi, :slow] setup=[Phonon, PhononLocal, TestCases] begin
    PhononLocal.test_frequencies(TestCases.aluminium_primitive)
end
