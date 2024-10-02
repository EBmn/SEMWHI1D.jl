using SEMWHI1D
using Documenter

DocMeta.setdocmeta!(SEMWHI1D, :DocTestSetup, :(using SEMWHI1D); recursive=true)

makedocs(;
    modules=[SEMWHI1D],
    authors="EBmn <elliotbckmn@gmail.com> and contributors",
    sitename="SEMWHI1D.jl",
    format=Documenter.HTML(;
        canonical="https://EBmn.github.io/SEMWHI1D.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/EBmn/SEMWHI1D.jl",
    devbranch="main",
)
