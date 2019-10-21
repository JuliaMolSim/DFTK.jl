using DFTK
using JLD
using LinearAlgebra: I
using NCDatasets
using PyCall
using Test
import PhysicalConstants.CODATA2018: a_0
abilab = pyimport("abipy.abilab")
abidata = pyimport("abipy.data")


ÅtoBohr = (1 / (a_0 * 1e10)).val


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
    JLD.save(joinpath(outdir, "extra.jld"), "extra", infile.extra)
    infile.extra = nothing

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

function load_reference(folder)
    ene = Dataset(joinpath(folder, "out_EIG.nc"))
    gsr = Dataset(joinpath(folder, "out_GSR.nc"))

    n_kpoints = size(gsr["reduced_coordinates_of_kpoints"], 2)
    bands = Vector{Vector{Float64}}(undef, n_kpoints)
    for ik in 1:n_kpoints
        bands[ik] = Vector(ene["Eigenvalues"][:, ik, 1])
    end

    energies = Dict{Symbol, Float64}(
                     :Ewald => gsr["e_ewald"][:],
                     :PspCorrection => gsr["e_corepsp"][:],
                     :PotXC => gsr["e_xc"][:],
                     :Kinetic => gsr["e_kinetic"][:],
                     :PotHartree => gsr["e_hartree"][:],
                     :PotLocal => gsr["e_localpsp"][:],
                     :PotNonLocal => gsr["e_nonlocalpsp"][:],
               )

    (energies=energies, bands=bands)
end

function load_density(T, folder::AbstractString)
    # Untested

    gsr = Dataset(joinpath(folder, "out_DEN.nc"))
    basis = load_basis(T, folder)

    n_comp = size(gsr["density"], 5)
    n_real_cpx = size(gsr["density"], 1)
    @assert n_comp == 1
    @assert n_real_cpx == 1

    ρ_real = Array{complex(T)}(gsr["density"][1, :, :, :, 1])
    @assert size(basis.FFT) == size(ρ_real)

    DFTK.r_to_G(basis, ρ_real)
end

function load_composition(T, gsr, folder::AbstractString)
    extra = JLD.load(joinpath(folder, "extra.jld"))["extra"]

    composition = Dict{Species, Vector{Vec3{T}}}()
    n_species = length(gsr["atomic_numbers"])
    for ispec in 1:n_species
        symbol = strip(join(gsr["chemical_symbols"][:, ispec]))
        spec = Species(Int(gsr["atomic_numbers"][ispec]),
                       psp=load_psp(extra["pspmap"][symbol]))

        mask_species = findall(isequal(ispec), gsr["atom_species"][:])
        positions = gsr["reduced_atom_positions"][:, mask_species]
        composition[spec] = [Vec3{T}(positions[:, m]) for m in mask_species]
    end
    pairs(composition)
end
function load_composition(T, folder)
    load_composition(T, Dataset(joinpath(folder, "out_GSR.nc")), folder)
end


function load_model(T, gsr, folder::AbstractString)
    # Parse functional ...
    functional = Vector{Symbol}()
    ixc = Int(gsr["ixc"][:])
    if ixc == 0
        # No XC functional
    elseif ixc == 1 || ixc == -20
        push!(functional, :lda_xc_teter93)
    elseif ixc == -1007
        push!(functional, :lda_x, :lda_c_vwn)
    elseif ixc == -101130 || ixc == 11
        push!(functional, :gga_x_pbe, :gga_c_pbe)
    else
        error("Unknown Functional ixc==$ixc")
    end

    # Parse smearing-scheme ...
    smearing = strip(join(gsr["smearing_scheme"][:]))
    Tsmear = 0
    smearing_function = nothing
    if smearing == "none"
        # No smearing
    elseif smearing == "Fermi-Dirac"
        smearing_function = DFTK.smearing_fermi_dirac
    elseif smearing == "Methfessel and Paxton"
        smearing_function = DFTK.smearing_methfessel_paxton_1
    else
        error("Unknown Smearing scheme: $smearing")
    end
    smearing_function !== nothing && (Tsmear = gsr["smearing_width"][:])

    # Build model and discretise
    lattice = Mat3{T}(gsr["primitive_vectors"][:])
    composition = load_composition(T, gsr, folder)
    model = nothing
    if length(functional) > 0
        model = model_dft(Array{T}(lattice), functional, composition...)
    else
        model = model_reduced_hf(Array{T}(lattice), composition...)
    end

    model
end
function load_model(T, folder::AbstractString)
    load_model(T, Dataset(joinpath(folder, "out_GSR.nc")), folder)
end


function load_basis(T, gsr, folder::AbstractString)
    model = load_model(T, gsr, folder)
    composition = load_composition(T, gsr, folder)

    Ecut = gsr["kinetic_energy_cutoff"][:]
    lattice = Mat3{T}(gsr["primitive_vectors"][:])
    kweights = Vector{T}(gsr["kpoint_weights"][:])

    n_kpoints = size(gsr["reduced_coordinates_of_kpoints"], 2)
    kcoords = Vector{Vec3{T}}(undef, n_kpoints)
    for ik in 1:n_kpoints
        kcoords[ik] = Vec3{T}(gsr["reduced_coordinates_of_kpoints"][:, ik])
    end

    kgrid_size = Vector{Int}(gsr["monkhorst_pack_folding"])
    kcoords_new, ksymops = bzmesh_ir_wedge(kgrid_size, lattice, composition...)
    @assert kcoords_new ≈ kcoords

    fft_size = determine_grid_size(model, Ecut)
    PlaneWaveModel(model, fft_size, Ecut, kcoords, ksymops)
end
function load_basis(T, folder::AbstractString)
    load_basis(T, Dataset(joinpath(folder, "out_GSR.nc")), folder)
end


function test_folder(T, folder; scf_tol=1e-8, n_ignored=0, test_tol=1e-6)
    basis = load_basis(T, folder)
    composition = load_composition(T, folder)
    ref = load_reference(folder)
    n_bands = length(ref.bands[1])

    ρ0 = guess_gaussian_sad(basis, composition...)
    ham = Hamiltonian(basis, ρ0)
    scfres = self_consistent_field!(ham, n_bands, tol=scf_tol)

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
