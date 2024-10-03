

using LinearAlgebra
using Plots
using FastGaussQuadrature
using LinearAlgebra

include("Wave_SEM_1d.jl")



function main()

    ##########################
    #set up the time interval, the number of elements, the degree of the interpolation polynomials

    Tend = 10.0
    nsteps = 10000

    xl = 1
    xr = 7
    NumberOfNodes = 256
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

    epsilon = 0.1

    for k = 1:length(nodes)-1
        for i = 1:N-1

            x[(k-1)*(N-1) + i] = nodes[k] + (1 + quadpoints[i])*(nodes[k+1]-nodes[k])/2 
            
            
            #f is some square wave, for instance
            if (abs((k-1)*(N-1) + i - Int(floor(length(nodes)/2))) < 20)
                fVals[(k-1)*(N-1) + i] = 10
            end

            #u begins as a standard mollifier
            uStart[(k-1)*(N-1) + i] = 5*exp(-1/(1-min(1, (x[(k-1)*(N-1) + i] - nodes[Int(floor(length(nodes)/2))])^2/epsilon^2)))

        end
    end

    animate = true
    snapshotFrequency = 100

    SEM_wave_1d.simulate(nodes, nsteps, Tend, N, fVals, omega, uStart, animate, snapshotFrequency)
    
end
