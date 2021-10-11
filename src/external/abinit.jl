using PyCall
import JSON

"""
Run an SCF in ABINIT starting from the input file `infile`
represented as a `abipy.abilab.AbinitInput` python object.
Write the result to the `output` directory in ETSF Nanoquanta format
and return the result as an `EtsfFolder` object.
"""
function run_abinit_scf(infile::PyObject, outdir)
    abilab = pyimport("abipy.abilab")
    flowtk = pyimport("abipy.flowtk")

    # Check ABINIT can be found
    abierrors = abilab.abicheck()
    abierrors != "" && error("abilab.abicheck() returned errors:\n$abierrors")

    # Create rundir
    rundir = joinpath(outdir, "abinit")
    if isdir(rundir)
        rm(rundir, recursive=true)
    end
    mkdir(rundir)

    # Adjust common infile settings:
    infile.set_vars(
        paral_kgb=0,    # Parallelization over k-points and Bands
        iomode=3,       # Use NetCDF output
        istwfk="*1",    # Needed for extracting the wave function later
    )

    # Dump pspmap
    open(joinpath(outdir, "pspmap.json"), "w") do fp
        JSON.print(fp, infile.pspmap)
    end
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

    EtsfFolder(outdir)
end


"""
Run an SCF in ABINIT starting from a DFTK `Model` and some extra parameters.
Write the result to the `output` directory in ETSF Nanoquanta format
and return the `EtsfFolder` object.
"""
function run_abinit_scf(model::Model, outdir;
                        kgrid, abinitpseudos, Ecut, n_bands, tol=1e-6, kwargs...)
    abilab = pyimport("abipy.abilab")
    Ecut = austrip(Ecut)

    # Would be nice to generate the pseudofiles in ABINIT format on the fly,
    # but not for now. See the function psp10in  in 64_psp/m_psp_hgh.F90
    # for the parser, which is used in ABINIT for GTH and HGH files.
    structure = abilab.Structure.as_structure(pymatgen_structure(model))
    pseudos = pyimport("abipy.data").pseudos(abinitpseudos...)
    infile = abilab.AbinitInput(structure=structure, pseudos=pseudos)

    # Copy what can just be copied
    infile.set_kmesh(ngkpt=Vector{Int}(kgrid), shiftk=[0, 0, 0])
    infile.set_vars(
        ecut=Ecut,        # Hartree
        nband=n_bands,    # Number of bands
        tolvrs=tol,       # General tolerance settings
    )

    # Spin-polarization
    if model.spin_polarization == :spinless
        error("spin_polarization == spinless is not supported by abinit")
    elseif model.spin_polarization == :none
        infile.set_vars(nsppol=1, nspinor=1, nspden=1)
    elseif model.spin_polarization == :collinear
        infile.set_vars(nsppol=2, nspinor=1, nspden=2)
    elseif model.spin_polarization == :full
        infile.set_vars(nsppol=1, nspinor=2, nspden=4)
    end

    # Check all required terms are there
    if any(isnothing, indexin([Kinetic(), AtomicLocal(), AtomicNonlocal(),
                               Ewald(), PspCorrection(), Hartree()], model.term_types))
        error("run_abinit_scf only supports reduced Hartree-Fock or DFT models.")
    end

    # Parse XC term
    idcs_Xc = findall(t -> t isa Xc, model.term_types)
    @assert length(idcs_Xc) <= 1
    if isempty(idcs_Xc) || isempty(model.term_types[idcs_Xc[1]].functionals)
        infile.set_vars(ixc=0)  # Reduced HF
    else
        functionals = sort(model.term_types[idcs_Xc[1]].functionals)
        if functionals == [:lda_xc_teter93]
            infile.set_vars(ixc=1)
        elseif functionals == [:lda_c_vwn, :lda_x]
            infile.set_vars(ixc="-001007")
        elseif functionals == [:gga_c_pbe, :gga_x_pbe]
            infile.set_vars(ixc="-101130")  # Version implemented in libxc
            # infile.set_vars(ixc=11)       # Version implemented in ABINIT
            # NOTE: The results only agree to 7 digits or so on a few test calculations I did
        else
            error("Unknown functional combination: $functionals")
        end
    end

    # Spin-polarization
    if model.spin_polarization == :spinless
        error("spin_polarization == spinless is not supported by abinit")
    elseif model.spin_polarization == :none
        infile.set_vars(nsppol=1, nspinor=1, nspden=1)
    elseif model.spin_polarization == :collinear
        infile.set_vars(nsppol=2, nspinor=1, nspden=2)
    elseif model.spin_polarization == :full
        infile.set_vars(nsppol=1, nspinor=2, nspden=4)
    end

    # Parse occopt
    if isa(model.smearing, Smearing.None)
        infile.set_vars(occopt=1)
    elseif isa(model.smearing, Smearing.FermiDirac)
        infile.set_vars(occopt=3, tsmear=model.temperature)
    elseif isa(model.smearing, Smearing.MethfesselPaxton2)
        infile.set_vars(occopt=6, tsmear=model.temperature)
    elseif isa(model.smearing, Smearing.Gaussian)
        infile.set_vars(occopt=7, tsmear=model.temperature)
    else
        error("Smearing $(model.smearing) not implemented.")
    end

    # Add pspmap
    pspmap = Dict{Int, String}()
    for (element, positions) in model.atoms
        element isa ElementPsp || continue
        element.psp.identifier == "" && continue
        pspmap[element.Z] = element.psp.identifier
    end
    infile.pspmap = pspmap

    !isempty(kwargs) && infile.set_vars(;kwargs...)
    run_abinit_scf(infile, outdir)
end
