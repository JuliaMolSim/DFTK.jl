abstract type StateType end

#state struct only containing the density, suitable for Anderson acceleration
mutable struct DensityType <: StateType 
    ρ
end

#state struct containing orbitals and occupations in addition, suitable for PCDIIS acceleration
mutable struct OrbitalType <: StateType
    ρ
    ψ
    occupation
end

#rearranges the entries of state and info to fit the inner part of the self_consistent_field() function
function rearrange(::Type{DensityType}, ρ, info)
    DensityType(ρ), info    
end

function rearrange(::Type{OrbitalType}, ρ, info)
    (; ρ, ψ, occupation) = merge((;ρ=ρ), info)
    info = Base.structdiff(info, NamedTuple{(:ρ, :ψ, :occupation)})
    OrbitalType(ρ, ψ, occupation), info
end

#return the content of the state struct as named tuple
function return_tuple(state::DensityType)
    (;ρ=state.ρ)
end

function return_tuple(state::OrbitalType)
    (;ρ=state.ρ, ψ=state.ψ, occupation=state.occupation)
end

#defining basic operations on the struct that only contains ρ
function Base.:-(state1::DensityType, state2::DensityType)
    DensityType(state1.ρ - state2.ρ)
end

function Base.:+(state1::DensityType, state2::DensityType)
    DensityType(state1.ρ + state2.ρ)
end

function Base.:-(state::DensityType, number::Number)
    DensityType(state.ρ .- number)
end

function Base.:+(state::DensityType, number::Number)
    DensityType(state.ρ .+ number)
end

function Base.:*(state::DensityType, number::Number)
    DensityType(state.ρ .* number)
end

LinearAlgebra.norm(x::DensityType, p::Real=2) = norm(x.ρ, p)

Base.vec(x::DensityType) = vec(x.ρ)

Base.:-(number::Number, state::DensityType) = number + (-state)
Base.:+(number::Number, state::DensityType) = state + number
Base.:*(number::Number, state::DensityType) = state * number
Base.:-(state::DensityType) = DensityType(-state.ρ) 
