# Parser functionality for folders adhering to the
# ETSF Nanoquanta file format, for details see http://www.etsf.eu/fileformats/

using NCDatasets
using JLD

struct EtsfFolder
    folder
    gsr
    den
    eig
    extra
end

"""
Initialise a EtsfFolder from the path to the folder which contains
the data in the ETSF Nanoquanta format.
"""
function EtsfFolder(folder::AbstractString)
    if !isfile(joinpath(folder, "out_GSR.nc"))
        error("Did not find file $folder/out_GSR.nc")
    end
    gsr = Dataset(joinpath(folder, "out_GSR.nc"))

    den = nothing
    eig = nothing
    extra = Dict{String, Any}()

    if isfile(joinpath(folder, "out_DEN.nc"))
        den = Dataset(joinpath(folder, "out_DEN.nc"))
    end
    if isfile(joinpath(folder, "out_EIG.nc"))
        eig = Dataset(joinpath(folder, "out_EIG.nc"))
    end
    if isfile(joinpath(folder, "extra.jld"))
        extra = JLD.load(joinpath(folder, "extra.jld"))["extra"]
    end

    EtsfFolder(folder, gsr, den, eig, extra)
end


"""
Load a DFTK-compatible compositon object from the ETSF folder.
Use the scalar type `T` to represent the data.
"""
function load_composition(T, folder::EtsfFolder)
    composition = Dict{Species, Vector{Vec3{T}}}()
    n_species = length(folder.gsr["atomic_numbers"])
    for ispec in 1:n_species
        symbol = strip(join(folder.gsr["chemical_symbols"][:, ispec]))
        spec = Species(Int(folder.gsr["atomic_numbers"][ispec]),
                       psp=load_psp(folder.extra["pspmap"][symbol]))

        mask_species = findall(isequal(ispec), folder.gsr["atom_species"][:])
        positions = folder.gsr["reduced_atom_positions"][:, mask_species]
        composition[spec] = [Vec3{T}(positions[:, m]) for m in mask_species]
    end
    pairs(composition)
end
load_composition(folder::EtsfFolder) = load_composition(Float64, folder)


"""
Load a DFTK-compatible model object from the ETSF folder.
Use the scalar type `T` to represent the data.
"""
function load_model(T, folder::EtsfFolder)
    # Parse functional ...
    functional = Vector{Symbol}()
    ixc = Int(folder.gsr["ixc"][:])
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
    smearing = strip(join(folder.gsr["smearing_scheme"][:]))
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
    smearing_function !== nothing && (Tsmear = folder.gsr["smearing_width"][:])

    # Build model and discretise
    lattice = Mat3{T}(folder.gsr["primitive_vectors"][:])
    composition = load_composition(T, folder)
    model = nothing
    if length(functional) > 0
        model = model_dft(Array{T}(lattice), functional, composition...;
                          smearing=smearing_function, temperature=Tsmear
                         )
    else
        model = model_reduced_hf(Array{T}(lattice), composition...;
                                 smearing=smearing_function, temperature=Tsmear
                                )
    end

    model
end
load_model(folder::EtsfFolder) = load_model(Float64, folder)


"""
Load a DFTK-compatible basis object from the ETSF folder.
Use the scalar type `T` to represent the data.
"""
function load_basis(T, folder::EtsfFolder)
    model = load_model(T, folder)
    composition = load_composition(T, folder)

    Ecut = folder.gsr["kinetic_energy_cutoff"][:]
    kweights = Vector{T}(folder.gsr["kpoint_weights"][:])

    n_kpoints = size(folder.gsr["reduced_coordinates_of_kpoints"], 2)
    kcoords = Vector{Vec3{T}}(undef, n_kpoints)
    for ik in 1:n_kpoints
        kcoords[ik] = Vec3{T}(folder.gsr["reduced_coordinates_of_kpoints"][:, ik])
    end

    kgrid_size = Vector{Int}(folder.gsr["monkhorst_pack_folding"])
    kcoords_new, ksymops = bzmesh_ir_wedge(kgrid_size, model.lattice, composition...)
    @assert kcoords_new ≈ kcoords

    PlaneWaveBasis(model, Ecut, kcoords, ksymops)
end
load_basis(folder::EtsfFolder) = load_basis(Float64, folder)


"""
Load a DFTK-compatible density object from the ETSF folder.
Use the scalar type `T` to represent the data.
"""
function load_density(T, folder::EtsfFolder)
    # So far this function is untested
    basis = load_basis(T, folder)

    n_comp = size(folder.den["density"], 5)
    n_real_cpx = size(folder.den["density"], 1)
    @assert n_comp == 1
    @assert n_real_cpx == 1

    ρ_real = Array{Complex{T}}(folder.den["density"][1, :, :, :, 1])
    @assert basis.fft_size == size(ρ_real)

    r_to_G(basis, ρ_real)
end
load_density(folder::EtsfFolder) = load_density(Float64, folder)
