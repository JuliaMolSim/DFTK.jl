using DFTK
using JLD
using LinearAlgebra: I
using PyCall
using Test
import PhysicalConstants.CODATA2018: a_0
abilab = pyimport("abipy.abilab")
abidata = pyimport("abipy.data")


const ÅtoBohr = (1 / (a_0 * 1e10)).val


"""Build H2 molecule with bond distance `r` in a cubic cell sized `a`"""
function build_hydrogen_structure(r, a=20)
    lattice = (a / ÅtoBohr) * Matrix(I, 3, 3)
    abilab.Structure(lattice', ["H", "H"], [[-r/2a, 0, 0], [r/2a, 0, 0]])
end


function build_silicon_structure()
    a = 5.431020504 * ÅtoBohr
    lattice = a / 2 * [[0 1 1]
                       [1 0 1]
                       [1 1 0]]

    # Note: Unit conversion because abilab insists on using Ångström
    abilab.Structure(lattice' / ÅtoBohr, ["Si", "Si"], [ones(3)/8, -ones(3)/8])
end

function build_magnesium_structure()
    a = 1.5970245  # Ångström
    b = 2.766127574932283  # Ångström
    c = 5.171978  # Ångström
    lattice = [[a a 0]; [-b b 0]; [0 0 c]]
    abilab.Structure(lattice', ["Mg", "Mg"],
                     [[2/3, 1/3, 1/4], [1/3, 2/3, 3/4]])
end

function build_graphite_structure()
    # Note: This is not exactly the minimum-energy structure
    a = 1.228  # Ångström
    b = 2.12695839  # Ångström
    c = 7  # Ångström
    lattice = [[a a 0]; [-b b 0]; [0 0 c]]
    abilab.Structure(lattice', ["C", "C", "C", "C"],
                     [[0, 0, 1/4], [0, 0, 3/4],
                      [1/3, 2/3, 1/4], [2/3, 1/3, 3/4]])
end

build_aluminium_structure() = abilab.Structure.fcc(7.6, ["Al"], units="bohr")

function run_ABINIT_scf(infile, outdir)
    flowtk = pyimport("abipy.flowtk")
    abilab.abicheck()

    # Create rundir
    rundir = joinpath(outdir, "abinit")
    if isdir(rundir)
        rm(rundir, recursive=true)
    end
    mkdir(rundir)

    # Adjust common infile settings:
    infile.set_vars(
        paral_kgb=0,    # Parallelisation over k-Points and Bands
        iomode=3,       # Use NetCDF output
        iscf=3,         # Anderson mixing instead of minimisation
        istwfk="*1",    # Needed for extracting the wave function later
    )

    # Dump extra data
    dump_extra(infile, outdir)
    infile.extra = nothing  # Remove extra field (causes problems below)

    # Create flow and run it
    flow = flowtk.Flow(rundir, flowtk.TaskManager.from_user_config())
    work = flowtk.Work()
    scf_task = work.register_scf_task(infile)
    flow.register_work(work)
    flow.allocate()
    flow.make_scheduler().start()

    if !flow.all_ok
        @warn "Flow not all_ok ... check input and output files"
    end

    for file in collect(scf_task.out_files())
        if endswith(file, ".nc")
            cp(file, joinpath(outdir, basename(file)), force=true)
        end
    end
end

function dump_extra(infile, outdir)
    JLD.save(joinpath(outdir, "extra.jld"), "extra", infile.extra)
end

function load_reference(folder::EtsfFolder)
    n_kpoints = size(folder.gsr["reduced_coordinates_of_kpoints"], 2)
    bands = Vector{Vector{Float64}}(undef, n_kpoints)
    for ik in 1:n_kpoints
        bands[ik] = Vector(folder.eig["Eigenvalues"][:, ik, 1])
    end

    energies = Dict{Symbol, Float64}(
                     :Ewald => folder.gsr["e_ewald"][:],
                     :PspCorrection => folder.gsr["e_corepsp"][:],
                     :PotXC => folder.gsr["e_xc"][:],
                     :Kinetic => folder.gsr["e_kinetic"][:],
                     :PotHartree => folder.gsr["e_hartree"][:],
                     :PotLocal => folder.gsr["e_localpsp"][:],
                     :PotNonLocal => folder.gsr["e_nonlocalpsp"][:],
               )

    (energies=energies, bands=bands)
end


function test_folder(T, folder; scf_tol=1e-8, n_ignored=0, test_tol=1e-6)
    @testset "$folder" begin
        etsf = EtsfFolder(folder)

        basis = load_basis(T, etsf)
        composition = load_composition(T, etsf)
        ref = load_reference(etsf)
        n_bands = length(ref.bands[1])

        ρ0 = guess_density(basis, composition...)
        ham = Hamiltonian(basis, ρ0)
        scfres = self_consistent_field(ham, n_bands, tol=scf_tol)

        energies = scfres.energies
        energies[:Ewald] = energy_nuclear_ewald(basis.model.lattice, composition...)
        energies[:PspCorrection] = energy_nuclear_psp_correction(basis.model.lattice,
                                                                 composition...)
        println("etot    ", sum(values(energies)) - sum(values(ref.energies)))

        for ik in 1:length(basis.kpoints)
            @test eltype(scfres.orben[ik]) == T
            @test eltype(scfres.Psi[ik]) == Complex{T}
            println(ik, "  ", abs.(scfres.orben[ik] - ref.bands[ik]))
        end
        for ik in 1:length(basis.kpoints)
            # Ignore last few bands, because these eigenvalues are hardest to converge
            # and typically a bit random and unstable in the LOBPCG
            diff = abs.(scfres.orben[ik] - ref.bands[ik])
            @test maximum(diff[1:n_bands - n_ignored]) < test_tol
        end
        for (key, value) in pairs(energies)
            if haskey(ref.energies, key)
                @test value ≈ ref.energies[key] atol=5test_tol
            end
        end
        @test sum(values(energies)) ≈ sum(values(ref.energies)) atol=test_tol
    end
end
