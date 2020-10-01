# Parser functionality for folders adhering to the
# ETSF Nanoquanta file format, for details see http://www.etsf.eu/fileformats/

using NCDatasets
using JSON

struct EtsfFolder
    folder
    gsr
    den
    eig
    pspmap
end

"""
Initialize a EtsfFolder from the path to the folder which contains
the data in the ETSF Nanoquanta format.
"""
function EtsfFolder(folder::AbstractString)
    if !isfile(joinpath(folder, "out_GSR.nc"))
        error("Did not find file $folder/out_GSR.nc")
    end
    gsr = Dataset(joinpath(folder, "out_GSR.nc"))

    den = nothing
    eig = nothing
    pspmap = Dict{Int, Any}()

    if isfile(joinpath(folder, "out_DEN.nc"))
        den = Dataset(joinpath(folder, "out_DEN.nc"))
    end
    if isfile(joinpath(folder, "out_EIG.nc"))
        eig = Dataset(joinpath(folder, "out_EIG.nc"))
    end
    if isfile(joinpath(folder, "pspmap.json"))
        dict = JSON.parsefile(joinpath(folder, "pspmap.json"))
        pspmap = Dict{Int, Any}(parse(Int, k) => dict[k] for k in keys(dict))
    end

    EtsfFolder(folder, gsr, den, eig, pspmap)
end


"""
Load a DFTK-compatible lattice object from the ETSF folder
"""
load_lattice(T, folder::EtsfFolder) = Mat3{T}(folder.gsr["primitive_vectors"][:])
load_lattice(folder; kwargs...) = load_lattice(Float64, folder; kwargs...)


"""
Load a DFTK-compatible atoms object from the ETSF folder.
Use the scalar type `T` to represent the data.
"""
function load_atoms(T, folder::EtsfFolder)
    atoms = Dict{ElementPsp, Vector{Vec3{T}}}()
    n_species = length(folder.gsr["atomic_numbers"])
    for ispec in 1:n_species
        atnum = Int(folder.gsr["atomic_numbers"][ispec])
        spec = ElementPsp(atnum, psp=load_psp(folder.pspmap[atnum]))

        mask_species = findall(isequal(ispec), folder.gsr["atom_species"][:])
        positions = folder.gsr["reduced_atom_positions"][:, mask_species]
        atoms[spec] = [Vec3{T}(positions[:, m]) for m in mask_species]
    end
    collect(pairs(atoms))
end
load_atoms(folder; kwargs...) = load_atoms(Float64, folder; kwargs...)


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
        smearing_function = Smearing.FermiDirac()
    elseif smearing == "Methfessel and Paxton"
        smearing_function = Smearing.MethfesselPaxton1()
    elseif smearing == "gaussian"
        smearing_function = Smearing.Gaussian()
    else
        error("Unknown Smearing scheme: $smearing")
    end
    smearing_function !== nothing && (Tsmear = folder.gsr["smearing_width"][:])

    n_spin = size(folder.gsr["eigenvalues"], 3)
    spin_polarization = n_spin == 2 ? :collinear : :none

    # Build model
    lattice = load_lattice(T, folder)
    atoms = load_atoms(T, folder)
    model_DFT(Array{T}(lattice), atoms, functional;
              smearing=smearing_function, temperature=Tsmear,
              spin_polarization=spin_polarization)
end
load_model(folder; kwargs...) = load_model(Float64, folder; kwargs...)


"""
Load a DFTK-compatible basis object from the ETSF folder.
Use the scalar type `T` to represent the data.
"""
function load_basis(T, folder::EtsfFolder)
    model = load_model(T, folder)
    atoms = load_atoms(T, folder)

    Ecut = folder.gsr["kinetic_energy_cutoff"][:]
    kcoords = Vec3{T}.(eachcol(folder.gsr["reduced_coordinates_of_kpoints"]))

    # Try to determine whether this is a shifted kpoint mesh or not
    ksmallest = sort(filter(k -> all(k .> 0), kcoords), by=norm)[1]
    kshift = [(abs(k) > 10eps(T)) ? 0.5 : 0.0 for k in ksmallest]

    kgrid = Vector{Int}(folder.gsr["monkhorst_pack_folding"])
    kcoords_new, ksymops, symops = bzmesh_ir_wedge(kgrid, model.symops, kshift=kshift)
    @assert kcoords_new ≈ kcoords

    PlaneWaveBasis(model, Ecut, kcoords, ksymops)
end
load_basis(folder; kwargs...) = load_basis(Float64, folder; kwargs...)


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

    from_real(basis, ρ_real)
end
load_density(folder; kwargs...) = load_density(Float64, folder; kwargs...)
