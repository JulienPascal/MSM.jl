using Documenter, MSM

makedocs(
         sitename = "MSM.jl",
         modules  = [MSM],
         authors = "Julien Pascal",
    pages =["Home" => "index.md",
        "Installation" => "installation.md",
        "Getting started" => "gettingstarted.md",
        ],

)
