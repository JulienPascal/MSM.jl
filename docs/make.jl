using Documenter
using MSM
using DataStructures
using OrderedCollections
using Random
using Distributions
using Statistics
using LinearAlgebra
using Plots
Random.seed!(1234)  #for replicability reasons

makedocs(
         sitename = "MSM.jl",
         modules  = [MSM],
         authors = "Julien Pascal",
    pages =["Home" => "index.md",
        "Installation" => "installation.md",
        "Getting started" => "gettingstarted.md",
        "Functions and Types" => "functions.md",
        "References" => "references.md",
        ],

)

deploydocs(
    repo = "github.com/JulienPascal/MSM.jl.git",
    devbranch = "main",
)
