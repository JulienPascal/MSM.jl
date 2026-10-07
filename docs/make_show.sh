#!/bin/bash
# Build the documentation, then serve it locally (show.jl needs LiveServer.jl in your default environment)
julia --project=. make.jl
julia show.jl
