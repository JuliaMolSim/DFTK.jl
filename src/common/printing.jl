# Internal helper functions for the show output of our structs.

const SHOWINDENTION = 4  # Number of spaces used for each indention level

function showfield(io::IO, key; indention=SHOWINDENTION)
    indentstr = " " ^ indention
    @printf io "%s%-20s %s " indentstr key (isempty(key) ? " " : ":")
end

function showfield(io::IO, key, value...; kwargs...)
    showfield(io, key; kwargs...)
    print(io, value...)
end

function showfieldln(io::IO, args...; kwargs...)
    showfield(io, args...; kwargs...)
    println(io)
end
