
using LinearAlgebra
using Plots
using FastGaussQuadrature
using LinearAlgebra
using TimerOutputs


include("Wave_SEM_1d.jl")


function main()

    #to = TimerOutput()


    ##########################
    #set up the time interval, the number of elements, the degree of the interpolation polynomials

    Tend = 10.0
    nsteps = 10000

    xl = 0
    xr = 1
    NumberOfNodes = 5
    nodes = zeros(NumberOfNodes)
    
    range = LinRange(xl, xr, NumberOfNodes)
    
    for i = 1:NumberOfNodes
        nodes[i] = range[i]
    end

    N = 5 #number of points we interpolate in for each element
    

    #set up some inital conditions and the values of the driving term f
    quadpoints, ~ = gausslobatto(N)
    totalPointCount = (length(nodes)-1) * (N-1) + 1

    x = zeros(totalPointCount)
    fVals = zeros(totalPointCount)
    uStart = zeros(totalPointCount)
    omega = 2.0

    simul = SEM_wave_1d.SEM_Wave(nodes, nsteps, Tend, N, fVals, omega)



    #@timeit to "Laplacian Computation" valueMatrix = SEM_wave_1d.LaplaceMMS(simul, 1e-8, "polynomial") 

    valueMatrix = SEM_wave_1d.LaplaceMMS(simul, 1e-8, "polynomial") 

    x = zeros((length(simul.nodes)-1) * (simul.PointsPerElement-1) + 1)

    for k = 1:length(simul.nodes)-1
        for i = 1:simul.PointsPerElement-1

            x[(k-1)*(simul.PointsPerElement-1) + i] = simul.nodes[k] + (1 + simul.QuadPoints[i])*(simul.nodes[k+1]-simul.nodes[k])/2 
            
        end
    end


    



    plt = plot(x[2:end-1], [valueMatrix[1] valueMatrix[2]], xlims=(x[2],x[end-1]), ylims=(-maximum(maximum(valueMatrix)), maximum(maximum(valueMatrix))), label = ["using LaplaceCalculation" "computed symbolically"])
    #plt = plot(x[2:end-1], abs.(valueMatrix[1] - valueMatrix[2]), xlims=(x[2],x[end-1]))#, ylims=(-10, 15), label = ["using LaplaceCalculation" "computed symbolically"])
    #plt = plot(x[2:end-1], [valueMatrix[1] valueMatrix[2]], xlims=(x[2],x[end-1]))
    plot!(x[2:end-1], zeros(length(x[2:end-1])), seriestype=:scatter, ms=0.5, label = "quadrature points")    

    savefig(plt, "testplot.png")



    println(maximum(abs.(valueMatrix[1] - valueMatrix[2])))

    #show(to)

end
