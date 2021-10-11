# Parser functionality for folders adhering to the
# ETSF Nanoquanta file format, for details see http://www.etsf.eu/fileformats/

export EtsfFolder

import JSON

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


load_lattice(T, folder::EtsfFolder) = Mat3{T}(folder.gsr["primitive_vectors"][:])


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


"""
Load a DFTK-compatible model object from the ETSF folder.
Use the scalar type `T` to represent the data.
"""
function load_model(T, folder::EtsfFolder; magnetic_moments=[])
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
    if spin_polarization == :collinear && isempty(magnetic_moments)
        @warn("load_basis will most likely fail for collinear spin if magnetic_moments " *
              "not explicitly specified.")
    end

    # Build model
    lattice = load_lattice(T, folder)
    atoms = load_atoms(T, folder)
    model_DFT(Array{T}(lattice), atoms, functional;
              smearing=smearing_function, temperature=Tsmear,
              spin_polarization=spin_polarization, magnetic_moments=magnetic_moments)
end

"""
Load a DFTK-compatible basis object from the ETSF folder.
Use the scalar type `T` to represent the data.
"""
function load_basis(T, folder::EtsfFolder; magnetic_moments=[])
    model = load_model(T, folder, magnetic_moments=magnetic_moments)
    atoms = load_atoms(T, folder)

    Ecut = folder.gsr["kinetic_energy_cutoff"][:]
    kcoords = Vec3{T}.(eachcol(folder.gsr["reduced_coordinates_of_kpoints"]))

    # Try to determine whether this is a shifted k-point mesh or not
    if length(kcoords) > 1
        ksmallest = sort(filter(k -> all(k .≥ 0), kcoords), by=norm)[1]
    else
        ksmallest = kcoords[1]
    end
    kshift = [(abs(k) > 10eps(T)) ? 0.5 : 0.0 for k in ksmallest]

    kgrid = Vector{Int}(folder.gsr["monkhorst_pack_folding"])
    kcoords_new, ksymops, _ = bzmesh_ir_wedge(kgrid, model.symmetries, kshift=kshift)
    @assert kcoords_new ≈ normalize_kpoint_coordinate.(kcoords)

    fft_size = size(folder.den["density"])[2:4]
    PlaneWaveBasis(model, Ecut, kcoords, ksymops, fft_size=fft_size,
                   kgrid=kgrid, kshift=kshift)
end


"""
Load a DFTK-compatible density object from the ETSF folder.
Use the scalar type `T` to represent the data.
"""
function load_density(T, folder::EtsfFolder; magnetic_moments=[])
    basis = load_basis(T, folder, magnetic_moments=magnetic_moments)

    n_real_cpx = size(folder.den["density"], 1)
    @assert n_real_cpx == 1

    n_spin = size(folder.den["density"], 5)
    @assert n_spin in (1, 2)

    if n_spin == 1
        @assert basis.model.spin_polarization == :none
        ρtot_real = Array{T}(folder.den["density"][1, :, :, :, 1])
        @assert basis.fft_size == size(ρ_real)

        return ρ_from_total_and_spin(ρtot_real)
    else
        @assert basis.model.spin_polarization == :collinear
        ρtot_real = Array{T}(folder.den["density"][1, :, :, :, 1])
        ρα_real = Array{T}(folder.den["density"][1, :, :, :, 2])
        @assert basis.fft_size == size(ρtot_real)
        @assert basis.fft_size == size(ρα_real)

        return ρ_from_total_and_spin(ρtot_real, 2ρα_real - ρtot_real)
    end
end
