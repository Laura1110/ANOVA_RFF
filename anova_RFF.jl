module anova_RFF

using LinearAlgebra, IterativeSolvers, LinearMaps, Distributed, SpecialFunctions

bases = ["exp", "cos"]
types = Dict(
    "exp" => ComplexF64,
    "cos" => Float64,
    "sin" => Float64,
)
vtypes = Dict(
    "exp" => Vector{ComplexF64},
    "cos" => Vector{Float64},
    "sin" => Vector{Float64},
)



include("algs_RFF.jl")
end # module