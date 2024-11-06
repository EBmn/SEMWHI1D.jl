


module SEM_Wave_1d

using LinearAlgebra
using Plots
using FastGaussQuadrature
using Polynomials
using TimerOutputs

include("MMS.jl")
using .MMS


export SEM_Wave

mutable struct SEM_Wave
    #stores data for SEM approximation of the wave equation \frac{\partial^2 u}{\partial t^2} = \frac{\partial^2 u}{\partial x^2} - f(x)\cos(\omega t)
    #Dirichlet boundary conditions are used

    nodes::Vector{Float64}          #the endpoints of the elements
    nsteps::Int64                   #number of steps we make in time
    Tend::Float64                   #the endpoint of our interval in time
    timestep::Float64               #size of the timesteps (= Tend/nsteps)
    PointsPerElement::Int64         #determines the degree of the Lagrange polynomials 
    uPrev::Vector{Float64}          #value of the displacement in the previous timepoint
    uNow::Vector{Float64}           #value of the displacement in the current timepoint
    uNext::Vector{Float64}          #value of the displacement in the next timepoint

    fVals::Vector{Float64}          #values of the driving term of the equation
    omega::Float64                  #frequency parameter omega
    QuadPoints::Vector{Float64}     #the points in [-1, 1] we use in our Gauss-Lobatto quadrature
    QuadWeights::Vector{Float64}    #the weights corresponding to the points in QuadPoints
    G::Array{Float64}               #a matrix required for the SEM method: G_{jm} = \sum_{l=0}^{N} w_l l'_m(\xi_l) l'_j(\xi_l); w_l \in QuadWeights, \xi_l \in QuadPoints, l_j's are the Lagrange polynomials
    inverseM::Vector{Float64}       #the inverse of the mass matrix, which is diagonal => stored as vector
    useMMS::Bool
    MMS_j::MMS_jet



    function SEM_Wave(
        nodes::Vector{Float64}, 
        nsteps::Int64,
        Tend::Float64,
        PointsPerElement::Int64,
        fVals::Vector{Float64}, 
        omega::Float64
        )
        
        QuadPoints, QuadWeights = gausslobatto(PointsPerElement)
        uPrev = zeros((length(nodes)-1)*(PointsPerElement-1) + 1)
        uNow = zeros((length(nodes)-1)*(PointsPerElement-1) + 1)
        uNext = zeros((length(nodes)-1)*(PointsPerElement-1) + 1)

        useMMS = false

        timestep = Tend/nsteps

        #make G and the inverse of the mass matrix
        G = ConstructG(QuadPoints, QuadWeights)
        inverseM = ConstructInverseM(nodes, QuadWeights)

        #change some coefficients in the MMS
        MMS_j = MMS.MMS_jet(1, 2)
        MMS_j.coeff[1, 2] = 2.0
        MMS_j.coeff[2, 1] = exp(1)
        MMS_j.coeff[2, 3] = -2
        MMS_j.coeff[1, 3] = (1 + sqrt(5))/2

            new(nodes, nsteps, Tend, timestep, PointsPerElement,
            uPrev, uNow, uNext, fVals, omega, QuadPoints, QuadWeights, G, inverseM, useMMS, MMS_j)
        
    end

end

function makeStep2!(simul::SEM_Wave, stepnumber::Int64)
    

    simul.uNext = 2*simul.uNow - simul.uPrev + 
                    simul.timestep^2 * LaplaceCalculation3(simul, simul.uNow) + 
                    simul.timestep^2 * ForcingTerm2(simul, stepnumber)        
    
    ###########
    #enforce Dirichlet boundary conditions
    simul.uNext[1] = 0
    simul.uNext[end] = 0
    
    #=
    if simul.useMMS
        simul.uNext[1] = 
        simul.uNext[end] = 
    end 
    =#
    


    #update the variables
    simul.uPrev = simul.uNow
    simul.uNow = simul.uNext

end

function LagrangeDer(points, i, j, h)
    #approximates the derivative of the ith Lagrange polynomial in the jth element of points
    #uses a central difference with step size h
    a = 1
    b = 1
    c = 1

    #evaluates \frac{l_i(points[j] + h) - l_i(points[j] - h)}{2h} using the definition of l_i
    for k = 1:length(points)
        if k != i
            a = a*(points[j] + h - points[k])
            b = b*(points[j] - h - points[k])
            c = c*(points[i] - points[k])
        end
    end

    return (a - b)/(2*h*c)

end

function simulate(nodes, nsteps, Tend, N, fVals, omega, uStart, animate = false, snapshotFrequency = 100, plotHeight = 10.0, animationName = "test.gif")
    #approximates a solution to the wave equation using SEM_Wave
    #creates a gif named animationName if animate == true
    #snapshotFrequency decides how many steps are made before a frame is saved in the gif
    #the y-axis shown in the gif is [-plotHeight, plotheight]


    #create a SEM_Wave with the specified parameter values
    simul = SEM_wave_1d.SEM_Wave(nodes, nsteps, Tend, N, fVals, omega)
    
    #################
    #take the first step here

    #stores the x-coordinates of the points of our interval
    x = zeros((length(nodes)-1) * (N-1) + 1)


    #puts the correct values in x and uNow 
    for k = 1:length(nodes)-1
        for i = 1:N-1

            x[(k-1)*(N-1) + i] = nodes[k] + (1 + simul.QuadPoints[i])*(nodes[k+1]-nodes[k])/2 
            
            simul.uNow[(k-1)*(N-1) + i] = uStart[(k-1)*(N-1) + i]
            
        end
    end
    
    x[end] = nodes[end]
    

    #create an Animation if one is asked for
    if (animate == true)
       
        anim = Animation()

        plot(x, simul.uNow, xlims=(nodes[1],nodes[end]), ylims=(-plotHeight, plotHeight))
        frame(anim)    
    
    end
    
    
    #make the steps
    for n=1:nsteps

        SEM_wave_1d.makeStep!(simul, n)

        #This is where you would store uNow for later if you want them remembered.
        

        #making the Animation is very slow compared to the actual calculations made here.
        #store a frame in our Animation once every snapshotFrequency steps
        if (animate == true && n%snapshotFrequency == 0)
            plot(x, simul.uNow, xlims=(nodes[1],nodes[end]), ylims=(-plotHeight, plotHeight))
            frame(anim)
            println(n)
        end

            
    end

    #finally make and store the gif
    if (animate == true)

        gif(anim, animationName, fps=40) #if you need the gif to run a different framerate, change the value of fps

    end

end

function simulate2(nodes, nsteps, Tend, N, fVals, omega, uStart, uStartDer, animate = false, snapshotFrequency = 100, plotHeight = 10.0, animationName = "test.gif")
    #approximates a solution to the wave equation using SEM_Wave
    #creates a gif named animationName if animate == true
    #snapshotFrequency decides how many steps are made before a frame is saved in the gif
    #the y-axis shown in the gif is [-plotHeight, plotheight]

    #create a SEM_Wave with the specified parameter values
    simul = SEM_wave_1d.SEM_Wave(nodes, nsteps, Tend, N, fVals, omega)

    initialise!(simul, uStart, uStartDer)

    #stores the x-coordinates of the points of our interval
    x = zeros((length(nodes)-1) * (N-1) + 1)

    #puts the correct values in x
    for k = 1:length(nodes)-1
        for i = 1:N-1
            x[(k-1)*(N-1) + i] = nodes[k] + (1 + simul.QuadPoints[i])*(nodes[k+1]-nodes[k])/2 
        end
    end
    
    x[end] = nodes[end]
    
    #create an Animation if one is asked for
    if (animate == true)
       
        anim = Animation()

        plot(x, simul.uNow, xlims=(nodes[1],nodes[end]), ylims=(-plotHeight, plotHeight))
        frame(anim)    
    
    end
    
    
    #make the steps
    for n=1:nsteps

        SEM_wave_1d.makeStep2!(simul, n)

        #This is where you would store uNow for later if you want them remembered.
        

        #making the Animation is very slow compared to the actual calculations made here.
        #store a frame in our Animation once every snapshotFrequency steps
        if (animate == true && n%snapshotFrequency == 0)
            plot(x, simul.uNow, xlims=(nodes[1],nodes[end]), ylims=(-plotHeight, plotHeight))
            frame(anim)
            println(n)
        end

            
    end

    #finally make and store the gif
    if (animate == true)

        gif(anim, animationName, fps=40) #if you need the gif to run a different framerate, change the value of fps

    end

end


function LaplaceCalculation3(simul::SEM_Wave, u::Vector{Float64})

    #to = TimerOutput()
    N = simul.PointsPerElement     #number of quadrature points
    K = length(simul.nodes)-1      #number of elements
    degreesOfFreedom = K*(N-1) + 1
   
    laplaceVals = zeros(degreesOfFreedom)

    u_k = zeros(N)
    v_k = zeros(N)
    G = simul.G

    for k = 1:K

        #=
        @timeit to "delta_x_k" delta_x_k = simul.nodes[k+1] - simul.nodes[k]

        @timeit to "get" u_k .= getDegreesOfFreedom(simul, k, u)

        #@timeit to "v_k" v_k .= (2/delta_x_k) * (simul.G*u_k)

        @timeit to "v_k" mul!(v_k, simul.G, u_k, 2/delta_x_k, 0)

        @timeit to "set" setDegreesOfFreedom!(simul, k, laplaceVals, v_k, true)
        =#

        delta_x_k = simul.nodes[k+1] - simul.nodes[k]

        u_k .= getDegreesOfFreedom(simul, k, u)

        mul!(v_k, simul.G, u_k, 2/delta_x_k, 0)

        setDegreesOfFreedom!(simul, k, laplaceVals, v_k, true)

    end

    #show(to)

    laplaceVals = -laplaceVals.*simul.inverseM      #the matrix corresponds to -Laplace, so we need to change the sign

    return laplaceVals

end


function LaplaceCalculation2(simul::SEM_Wave, u::Vector{Float64})

    degreesOfFreedom = (length(simul.nodes)-1)*(simul.PointsPerElement-1) + 1
    laplaceVals = zeros(degreesOfFreedom)

    for k = 1:length(simul.nodes) - 1
        for j = 2:simul.PointsPerElement

            delta_x_k = simul.nodes[k+1] - simul.nodes[k]

            laplaceVal = 0

            #if we are considering a boundary point which is not simul.nodes[1] or simul.nodes[end]:
            if (j == simul.PointsPerElement && k != length(simul.nodes)-1)
        
                delta_x_kNext = simul.nodes[k+2] - simul.nodes[k+1] 

                #this sum now has to take both of the element it borders into account
                for m = 1:simul.PointsPerElement
                    laplaceVal = laplaceVal + u[(k-1)*(simul.PointsPerElement-1) + m]*simul.G[simul.PointsPerElement, m]/delta_x_k + u[k*(simul.PointsPerElement-1) + m]*simul.G[1, m]/delta_x_kNext
                end
        
                weightFactor = 1/(0.5*simul.QuadWeights[simul.PointsPerElement]*delta_x_k + 0.5*simul.QuadWeights[1]*delta_x_kNext)

                laplaceVal = -2*weightFactor*laplaceVal

                laplaceVals[(k-1)*(simul.PointsPerElement-1) + j] = laplaceVal

            else
            #if we are considering an interior point:

                for m = 1:simul.PointsPerElement
                    laplaceVal = laplaceVal + u[(k-1)*(simul.PointsPerElement-1) + m]*simul.G[j, m]
                end
        
                laplaceVal = -4/(simul.QuadWeights[j]*delta_x_k^2) * laplaceVal

                laplaceVals[(k-1)*(simul.PointsPerElement-1) + j] = laplaceVal

            end


        end
    end

    return laplaceVals

end


function LaplaceCalculation(simul::SEM_Wave, u::Vector{Float64}, j::Int64, k::Int64)
    #Calculates the term corresponding to the Laplacian in our SEM method
    #point number j in element number k
    #for boundary points we should always call this function with j = simul.PointsPerElement and k != length(simul.nodes)

    delta_x_k = simul.nodes[k+1] - simul.nodes[k]

    laplaceVal = 0

    #if we are considering a boundary point which is not simul.nodes[1] or simul.nodes[end]:
    if (j == simul.PointsPerElement && k != length(simul.nodes)-1)
        

        delta_x_kNext = simul.nodes[k+2] - simul.nodes[k+1] 

        #this sum now has to take both of the element it borders into account
        for m = 1:simul.PointsPerElement
            laplaceVal = laplaceVal + u[(k-1)*(simul.PointsPerElement-1) + m]*simul.G[simul.PointsPerElement, m]/delta_x_k + u[k*(simul.PointsPerElement-1) + m]*simul.G[1, m]/delta_x_kNext
        end
        
        weightFactor = 1/(0.5*simul.QuadWeights[simul.PointsPerElement]*delta_x_k + 0.5*simul.QuadWeights[1]*delta_x_kNext)

        laplaceVal = -2*weightFactor*laplaceVal

    else

        #if we are considering an interior point:

        for m = 1:simul.PointsPerElement
            laplaceVal = laplaceVal + u[(k-1)*(simul.PointsPerElement-1) + m]*simul.G[j, m]
        end
        
        laplaceVal = -4/(simul.QuadWeights[j]*delta_x_k^2) * laplaceVal

    end

    #if you attempt to use this function for calculating the Laplacian at the final endpoint
    #ideally some exception would be thrown here
    if (j == simul.PointsPerElement && k == length(simul.nodes)-1)
        println("Warning: LaplaceCalculation has been given the wrong index, corresponding to the boundary point of the final element!")
    end

    return laplaceVal

end


function ForcingTerm2(simul::SEM_Wave, stepnumber::Int64)

    N = simul.PointsPerElement     #number of quadrature points
    K = length(simul.nodes)-1      #number of elements
    degreesOfFreedom = K*(N-1) + 1
   
    forcingVals = zeros(degreesOfFreedom)

    F_k = zeros(N)
    vals_k = zeros(N)
    
    if simul.useMMS

        t = simul.timestep*stepnumber

        x_loc = zeros(N)

        z = (1 .+ simul.QuadPoints)

        for k = 1:K

            delta_x_k = simul.nodes[k+1] - simul.nodes[k]
            
            x_loc .= simul.nodes[k] .+ 0.5*z*delta_x_k

            for j = 1:N
                vals_k[j] = MMS.MMSfun(x_loc[j], t, 0, 2, simul.MMS_j) - MMS.MMSfun(x_loc[j], t, 2, 0, simul.MMS_j)
            end
            

            setDegreesOfFreedom!(simul, k, forcingVals, vals_k, false)

        end

    else

        for k = 1:K

            delta_x_k = simul.nodes[k+1] - simul.nodes[k]

            F_k .= getDegreesOfFreedom(simul, k, simul.fVals)

            vals_k = 0.5*delta_x_k * (simul.QuadWeights.*F_k)

            setDegreesOfFreedom!(simul, k, forcingVals, vals_k, true)

        end

        forcingVals = forcingVals.*simul.inverseM * cos(simul.omega*(stepnumber-1)*simul.timestep)

    end

    return forcingVals

end

function ForcingTerm(simul::SEM_Wave, j::Int64, k::Int64, stepnumber::Int64)

    delta_x_k = simul.nodes[k+1] - simul.nodes[k]
    output = 0
    
    #if we are considering a boundary point which is not simul.nodes[1] or simul.nodes[end]:
    if (j == simul.PointsPerElement && k != length(simul.nodes)-1)

        delta_x_kNext = simul.nodes[k+2] - simul.nodes[k+1]

        weightFactor = 1/(0.5*simul.QuadWeights[simul.PointsPerElement]*delta_x_k + 0.5*simul.QuadWeights[1]*delta_x_kNext)
        output = weightFactor*(0.5*simul.QuadWeights[simul.PointsPerElement]*delta_x_k + 
                               0.5*simul.QuadWeights[1]*delta_x_kNext)*
                               simul.fVals[(k-1)*(simul.PointsPerElement-1) + 1]*cos(simul.omega*(stepnumber-1)*simul.timestep)


    #else our indices j, k correspond to an interior point
    else

        output = simul.fVals[(k-1)*(simul.PointsPerElement-1) + j]*cos(simul.omega*(stepnumber-1)*simul.timestep)

    end


    #if you attempt to use this function for calculating the forcing term at the final endpoint
    #ideally some exception would be thrown here
    if (j == simul.PointsPerElement && k == length(simul.nodes)-1)
        println("Warning: ForcingTerm has been given the wrong index, corresponding to the boundary point of the final element!")
    end

    return output

end

function initialise!(simul::SEM_Wave, uStart::Vector{Float64}, uStartDer::Vector{Float64})
    
    simul.uNow = uStart

    simul.uPrev = uStart - simul.timestep*uStartDer + 0.5*simul.timestep^2*(LaplaceCalculation3(simul, uStart) + ForcingTerm2(simul, 1)) #stepnumber = 1 corresponds to t = 0

end

function ConstructG(QuadPoints::Vector{Float64}, QuadWeights::Vector{Float64})

    N = length(QuadWeights)

    D = zeros(N, N)

    weights = BarycentricWeights(QuadPoints)

    for i = 1:N
        for j = 1:N
            if (i != j)
                D[i, j] = (weights[j]/weights[i])*(1/(QuadPoints[i] - QuadPoints[j]))
                D[i, i] -= D[i, j]
            end
        end
    end

    return transpose(D)*(QuadWeights.*D)

end


function ConstructInverseM(nodes::Vector{Float64}, QuadWeights::Vector{Float64})

    #constructs the inverse of the mass matrix

    K = length(nodes)-1
    N = length(QuadWeights)

    inverseM = zeros(K*(N-1) + 1)
    
    for k = 1:K
            
        start = (k-1)*(N-1)

        delta_x_k = nodes[k+1] - nodes[k]

        inverseM[(start + 1):(start + N)] += QuadWeights*delta_x_k/2
        
    end

    inverseM = 1 ./inverseM

end


function BarycentricWeights(points::Vector{Float64})

    #returns the barycentric weights of a vector of points; used in calculating the derivative matrix of the Lagrange polynomials

    N = length(points)
    weights = ones(N)

    for i = 1:N
        for j = 1:N
            if (i != j)
                weights[j] *= 1/(points[j] - points[i])
            end
        end
    end

    return weights

end


function getDegreesOfFreedom(simul::SEM_Wave, k::Int64, u::Vector{Float64})
    start = (k-1)*(simul.PointsPerElement-1)
    return u[(start + 1):(start + simul.PointsPerElement)]

end


function setDegreesOfFreedom!(simul::SEM_Wave, k::Int64, v::Vector{Float64}, v_k::Vector{Float64}, add::Bool)
    
    start = (k-1)*(simul.PointsPerElement-1)

    if add
        #v[(start + 1):(start + simul.PointsPerElement)] .+= v_k
        view(v, (start + 1):(start + simul.PointsPerElement)) .+= v_k
    else
        v[(start + 1):(start + simul.PointsPerElement)] .= v_k
    end

end


function LaplaceMMS(simul::SEM_Wave, tol::Float64, type)

    acceptableAccuracy = true

    if type == "polynomial"

        #make a polynomial which we will test against, and calculate its derivative symbolically
        #testPolynomial = Polynomial([1, 2, 3, 3, 5, 6, -7, 3, 1, -2, 1])
        testPolynomial = Polynomial([0, 1, -1])^3 * Polynomial([0, 1])^4
        #testPolynomial = Polynomial([0, 1])
        testPolynomial_xx = derivative(derivative(testPolynomial))

        testPolynomialValues = zeros(length(simul.uNow))
        testPolynomial_xxValues = zeros(length(testPolynomialValues))

        #evaluate the polynomial for the appropriate x-values given by simul.nodes
        for k = 1:length(simul.nodes)-1
            for j = 1:simul.PointsPerElement-1
                testPolynomialValues[(k-1)*(simul.PointsPerElement-1) + j] = testPolynomial(simul.nodes[k] + (1 + simul.QuadPoints[j])*(simul.nodes[k+1]-simul.nodes[k])/2)
                testPolynomial_xxValues[(k-1)*(simul.PointsPerElement-1) + j] = testPolynomial_xx(simul.nodes[k] + (1 + simul.QuadPoints[j])*(simul.nodes[k+1]-simul.nodes[k])/2)
            end
        end        

        testPolynomialValues[end] = testPolynomial(simul.nodes[end])
        

        #calculate the laplacian of this polynomial using LaplaceCalculation
        laplaceValues = zeros(length(testPolynomialValues))

        #=
        for k = 1:length(simul.nodes)-1
            
            for j = 2:simul.PointsPerElement
                
                #this makes sure to exclude simul.nodes[1] or simul.nodes[end]
                if (k != length(simul.nodes)-1 || j != simul.PointsPerElement)
                    laplaceValues[(k-1)*(simul.PointsPerElement-1) + j] = LaplaceCalculation(simul, testPolynomialValues, j, k)
                end

            end

        end
        =#

        #laplaceValues = LaplaceCalculation2(simul, testPolynomialValues)
        laplaceValues = LaplaceCalculation3(simul, testPolynomialValues)

        #println(norm(laplaceValues[2:end-1] - laplaceValues3[2:end-1]))

        #compare with the value from LaplaceCalculation

        #laplaceValues[8] = 10^3
        
        #diff = norm(laplaceValues[2:length(laplaceValues)-1] - testPolynomial_xxValues[2 : length(laplaceValues)-1])
        diff = norm(laplaceValues[2:end-1] - testPolynomial_xxValues[2 : end-1])

        #println(laplaceValues[2:end-1] - testPolynomial_xxValues[2:end-1])

        if (diff > tol)
            acceptableAccuracy = false
        end

        
    end

    if (acceptableAccuracy)
        println("LaplaceCalculation passed the MMS test")
    else
        println("LaplaceCalculation did not pass the MMS test")
    end

    return [laplaceValues[2:end-1], testPolynomial_xxValues[2:end-1]]
    #return [laplaceValues[2:end-1], laplaceValues3[2:end-1]]
    
end


function MMS_old(simul::SEM_Wave, tol::Float64, type)

    acceptableAccuracy = true

    N = simul.PointsPerElement
    K = length(simul.nodes) - 1

    if type == "polynomial"

        #the forcing term will be g_tt - g_xx, where g = pol(x)cos(omega*t)
        p = 3
        q = 4
        pol = Polynomial([0, 1, -1])^p * Polynomial([0, 1])^q
        pol_xx = derivative(derivative(Polynomial([0, 1, -1])^p * Polynomial([0, 1])^q))
        polValues = zeros(length(simul.uNow))
        pol_xxValues = zeros(length(simul.uNow))
        
        #evaluate the polynomial and its derivative for the appropriate x-values given by simul.nodes
        for k = 1:K
            
            delta_x_k = simul.nodes[k+1]-simul.nodes[k]

            for j = 1:N-1
                polValues[(k-1)*(N-1) + j] = pol(simul.nodes[k] + (1 + simul.QuadPoints[j])*delta_x_k/2)
                pol_xxValues[(k-1)*(N-1) + j] = pol_xx(simul.nodes[k] + (1 + simul.QuadPoints[j])*delta_x_k/2)
            end

        end        

        polValues[end] = pol(simul.nodes[end])
        pol_xxValues[end] = pol_xx(simul.nodes[end])

        simulate2!(simul.nodes, simul.nsteps, simul.Tend, simul.PointsPerElement, simul.fVals, simul.omega, zeros(length(simul.uNow)))
        
        stopTime = simul.nsteps*simul.timestep

        diff =  norm(simul.uNow - polValues*cos(omega*stopTime))

        if (diff > tol)
            acceptableAccuracy = false
        end

    end

    if (acceptableAccuracy)
        println("Passed the MMS test")
    else
        println("Did not pass the MMS test")
    end

    return [simul.uNow,  polValues*cos(omega*stopTime)]
    
end



end