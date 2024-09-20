"""
MPI reduction opertations with custom types (i.e. anything that has not a MPI datatype equivalent)
are not available on aarch64. These are temprorary workarounds, where variables with custom types 
are broken down to standard types before communication, and recast to the initial types after.
This file was created by fixing all MPI errors encountered by running the tests on an ARM machine:
all sensible MPI reduction routines are implemented for each custom type causing an error.
"""

# Julia's Bool type has no direct equivalent MPI datatype => need integer conversion
function mpi_min(bool::Bool, comm::MPI.Comm)
    int = Int(bool)
    Bool(mpi_min(int, comm))
end

function mpi_max(bool::Bool, comm::MPI.Comm)
    int = Int(bool)
    Bool(mpi_max(int, comm))
end

# Vec3{T} must be cast to Vector{T} before MPI reduction
function mpi_sum!(arr::Vector{Vec3{T}}, comm::MPI.Comm) where{T}
    n = length(arr)
    new_arr = zeros(T, 3n)
    for i in 1:n
        new_arr[3(i-1)+1:3(i-1)+3] = @view arr[i][1:3]
    end
    mpi_sum!(new_arr, comm)
    for i in 1:n
	    arr[i] = Vec3{T}(@view new_arr[3(i-1)+1:3(i-1)+3])
    end
    arr
end

# ForwardDiff.Dual{T, U, V} and arrays of it must be cast to Vector{U} as well
# utility function to cast a Dual type to an array containing a value and the partial diffs
function dual_array(dual::ForwardDiff.Dual{T, U, V}) where{T, U, V}
    dual_array = zeros(U, ForwardDiff.npartials(dual)+1)
    dual_array[1] = ForwardDiff.value(dual)
    dual_array[2:end] = @view dual.partials[1:end]
    dual_array
end

# utility function that casts back an array to a Dual type, based on a template Dual
function new_dual(dual_array, template::ForwardDiff.Dual{T, U, V}) where{T, U, V}
    ForwardDiff.Dual{T}(dual_array[1], Tuple(@view dual_array[2:end]))
end

# MPI reductions of single ForwardDiff.Dual types
function mpi_sum(dual::ForwardDiff.Dual{T, U, V}, comm::MPI.Comm) where{T, U, V}
    arr = dual_array(dual)
    mpi_sum!(arr, comm)
    new_dual(arr, dual)
end

function mpi_min(dual::ForwardDiff.Dual{T, U, V}, comm::MPI.Comm) where{T, U, V}
    arr = dual_array(dual)
    mpi_min!(arr, comm)
    new_dual(arr, dual)
end

function mpi_max(dual::ForwardDiff.Dual{T, U, V}, comm::MPI.Comm) where{T, U, V}
    arr = dual_array(dual)
    mpi_max!(arr, comm)
    new_dual(arr, dual)
end

function mpi_mean(dual::ForwardDiff.Dual{T, U, V}, comm::MPI.Comm) where{T, U, V}
    arr = dual_array(dual)
    mpi_mean!(arr, comm)
    new_dual(arr, dual)
end

# MPI reductions of arrays of ForwardDiff.Dual types
function mpi_sum!(dual::Array{ForwardDiff.Dual{T, U, V}, N}, comm::MPI.Comm) where{T, U, V, N}
    array = Vector{U}([])
    lengths = []
    for i in 1:length(dual)
        tmp = dual_array(dual[i])
        append!(array, tmp)
        append!(lengths, length(tmp))
    end
    mpi_sum!(array, comm)
    offset = 0
    for i in 1:length(dual)
        view = @view array[offset+1:offset+lengths[i]]
        dual[i] = new_dual(view, dual[i])
        offset += lengths[i]
    end
    dual
end
