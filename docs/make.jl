using Documenter, MSM

makedocs(
         sitename = "MSM.jl",
         modules  = [MSM],
         authors = "Julien Pascal",
    pages =["Home" => "index.md",
        "Installation" => "installation.md",
        "Getting started" => "gettingstarted.md",
        "References" => "references.md",
        ],

)

deploydocs(
    repo = "github.com/JulienPascal/MSM.jl.git",
    devbranch = "main",
)
