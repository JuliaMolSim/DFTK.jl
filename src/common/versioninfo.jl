using PkgVersion


"""
    DFTK.versioninfo([io::IO=stdout])

Summary of version and configuration of DFTK and its key dependencies.
"""
function versioninfo(io::IO=stdout)
    indent = "  "
    println(io, "DFTK Version      ", PkgVersion.@Version)
    println(io, "Julia Version     ", VERSION)
    println(io, "FFTW.jl provider  ", FFTW.get_provider(), " v$(FFTW.version)")
    println(io)

    let versionstr = sprint(show, "text/plain", BLAS.get_config())
        println(io, "BLAS.get_config()")
        println(io, indent, replace(versionstr, "\n" => "\n" * indent), "\n")
    end

    let versionstr = sprint(MPI.versioninfo)
        println(io, "MPI.versioninfo()")
        println(io, indent, replace(versionstr, "\n" => "\n" * indent), "\n")
    end
end
