using PkgVersion


"""
    DFTK.versioninfo([io::IO=stdout])

Summary of version and configuration of DFTK and its key dependencies.
"""
function versioninfo(io::IO=stdout)
    indent = "  "
    println(io, "DFTK Version     ", PkgVersion.@Version)
    println(io, "Julia Version    ", VERSION)

    if !isdefined(MPI, :versioninfo)
        println(io, "MPI.jl Version:  ", PkgVersion.Version(MPI))
    else
        println(io)
        let versionstr = sprint(MPI.versioninfo)
            println(io, "MPI.versioninfo()")
            println(io, indent, replace(versionstr, "\n" => "\n" * indent))
        end
    end

    let versionstr = sprint(show, "text/plain", BLAS.get_config())
        println(io, "BLAS.get_config()")
        println(io, indent, replace(versionstr, "\n" => "\n" * indent), "\n")
    end
end
