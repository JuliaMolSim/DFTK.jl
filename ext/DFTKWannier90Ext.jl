module DFTKWannier90Ext

using DFTK
import wannier90_jll

"""
Run the Wannierization procedure with Wannier90.
"""
@DFTK.timing function run_wannier90(scfres;
        n_bands=scfres.n_bands_converge,
        n_wannier=n_bands,
        projections::AbstractVector=DFTK.default_wannier_centers(n_wannier),
        fileprefix=joinpath("wannier90", "wannier"),
        wannier_plot=false,
        kwargs...)

    prefix, dir = basename(fileprefix), dirname(fileprefix)

    # Prepare files
    DFTK.write_wannier90_files(scfres; n_bands, n_wannier, projections, fileprefix, wannier_plot, kwargs...) do
        wannier90_jll.wannier90(exe -> run(Cmd(`$exe -pp $prefix`; dir)))
        DFTK.read_w90_nnkp(fileprefix)
    end

    # Run Wannierisation procedure
    @DFTK.timing "Wannierization" wannier90_jll.wannier90(exe -> run(Cmd(`$exe $prefix`; dir)))
    fileprefix
end

end