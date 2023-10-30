"""
Use Wannier.jl to read a .win file and produce the nnkp data (i.e. the b-vectors used in the Mmn matrix).
"""
function get_nnkpt_from_wannier(fileprefix)
    model = Wannier.read_w90(fileprefix; amn=false, mmn=false, eig=false)
    bvectors = model.bvectors

    nnkpts = reduce(vcat,
        map(1:model.n_kpts) do ik
            zp = zip(bvectors.kpb_k[:, ik], eachcol(bvectors.kpb_b[:, :, ik]))
            map(zp) do (ikpb, G_shift)
                (ik, ikpb, G_shift=copy(G_shift))
            end
        end)

    (nntot=model.n_bvecs, nnkpts)
end

"""
Build a Wannier.jl model that can be used for Wannierization.
"""
@timing function get_wannier_model(scfres;
        n_bands=scfres.n_bands_converge,
        n_wannier=n_bands,
        projections::AbstractVector{<:WannierProjection}=default_wannier_centres(n_wannier),
        fileprefix=joinpath("wannierjl", "wannier"),
        wannier_plot=false,
        kwargs...)
    # Write the files
    write_wannier90_files(scfres; n_bands, n_wannier, projections, fileprefix, wannier_plot, kwargs...) do 
        get_nnkpt_from_wannier(fileprefix)
    end

    # Read Wannier.jl model
    Wannier.read_w90(fileprefix)
end