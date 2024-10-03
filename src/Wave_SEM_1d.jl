module SEM_wave_1d

using LinearAlgebra
using Plots
using FastGaussQuadrature

export SEM_Wave

#testcomment

mutable struct SEM_Wave
    #stores data for SEM approximation of the wave equation \frac{\partial u}{\partial t^2} = \frac{\partial u}{\partial x^2} - f(x)\cos(\omega t)
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
        G = zeros(PointsPerElement, PointsPerElement)

        timestep = Tend/nsteps


        #fill G with the correct elements
        for j = 1:PointsPerElement
            for m = 1:PointsPerElement
                for l = 1:PointsPerElement
                    G[j, m] = G[j, m] + QuadWeights[l]*LagrangeDer(QuadPoints, m, l, 0.000001)*LagrangeDer(QuadPoints, j, l, 0.000001)
                end
            end
        end

        #here you might want to check some stability condition

            new(nodes, nsteps, Tend, timestep, PointsPerElement,
            uPrev, uNow, uNext, fVals, omega, QuadPoints, QuadWeights, G)
        
    end

    
end



function makeStep!(simul::SEM_Wave, stepnumber::Int64)
    #this function makes a step in time and updates uNow and uPrev accordingly

    for k = 1:length(simul.nodes)-1 #for each element...

        if (stepnumber == 1) #the first step is different
            
            for j = 2:simul.PointsPerElement-1      #sort out the interior points
            
                sum = 0
                
            
                #this sum is needed for the SEM method
                for m = 1:simul.PointsPerElement
                    sum = sum + simul.uNow[(k-1)*(simul.PointsPerElement-1) + m]*simul.G[j, m]
                end
        
                #store values in uNext in accordance with our method
                simul.uNext[(k-1)*(simul.PointsPerElement-1) + j] = simul.uNow[j + (k-1)*(simul.PointsPerElement-1)] + 0.5*simul.timestep^2*(-simul.fVals[(k-1)*(simul.PointsPerElement-1) + j]*cos((stepnumber-1)*simul.timestep*simul.omega) - 4*sum/(simul.QuadWeights[j + 1]*(simul.nodes[k+1] - simul.nodes[k])^2))  
    
            end

            #and now the boundary elements: only the right boundary points from all but the final element need to be adjusted, due to the boundary conditions

            if k != (length(simul.nodes)-1)

                wFactor = 1/(0.5*simul.QuadWeights[simul.PointsPerElement]*(simul.nodes[k+1] - simul.nodes[k]) + 0.5*simul.QuadWeights[1]*(simul.nodes[k+2] - simul.nodes[k+1]))

                sum = 0

                for m = 1:simul.PointsPerElement
                    sum = sum + 2*simul.uNow[(k-1)*(simul.PointsPerElement-1) + m]*simul.G[simul.PointsPerElement, m]/(simul.nodes[k+1] - simul.nodes[k]) +  2*simul.uNow[k*(simul.PointsPerElement-1) + m]*simul.G[1, m]/(simul.nodes[k+2] - simul.nodes[k+1])
                end

                #store values in uNext in accordance with our method
                simul.uNext[k*(simul.PointsPerElement-1) + 1] = simul.uNow[k*(simul.PointsPerElement-1) + 1] - 
                                                                                            0.5*simul.timestep^2*wFactor*(sum - (0.5*simul.QuadWeights[1]*(simul.nodes[k+2] - simul.nodes[k+1]) - 0.5*simul.QuadWeights[simul.PointsPerElement]*(simul.nodes[k+1] - simul.nodes[k])*(-simul.fVals[(k-1)*(simul.PointsPerElement-1) + 1]*cos(simul.omega*(stepnumber-1)*simul.timestep))))
            end

        else

            for j = 2:simul.PointsPerElement-1      #sort out the interior points
            
                #this sum is needed for the SEM method
                sum = 0
                for m = 1:simul.PointsPerElement
                    sum = sum + simul.uNow[(k-1)*(simul.PointsPerElement-1) + m]*simul.G[j, m]
                end

                #store values in uNext in accordance with our method
                simul.uNext[(k-1)*(simul.PointsPerElement-1) + j] = 2*simul.uNow[j + (k-1)*(simul.PointsPerElement-1)] - simul.uPrev[j + (k-1)*(simul.PointsPerElement-1)] -
                                                                    4*simul.timestep^2/(simul.QuadWeights[j]*(simul.nodes[k+1] - simul.nodes[k])^2) * sum + simul.timestep^2*(simul.fVals[(k-1)*(simul.PointsPerElement-1) + j]*cos(simul.omega*(stepnumber-1)*simul.timestep))

            end

            #and now the boundary elements: only the right boundary points from all but the final element need to be adjusted

            if k != length(simul.nodes)-1
                
                wFactor = 1/(0.5*simul.QuadWeights[simul.PointsPerElement]*(simul.nodes[k+1] - simul.nodes[k]) + 0.5*simul.QuadWeights[1]*(simul.nodes[k+2] - simul.nodes[k+1]))

                #this sum is needed for the SEM method
                sum = 0
                for m = 1:simul.PointsPerElement
                    sum = sum + simul.uNow[(k-1)*(simul.PointsPerElement-1) + m]*simul.G[simul.PointsPerElement, m]/(simul.nodes[k+1] - simul.nodes[k]) +  simul.uNow[k*(simul.PointsPerElement-1) + m]*simul.G[1, m]/(simul.nodes[k+2] - simul.nodes[k+1])
                end

                #store values in uNext in accordance with our method
                simul.uNext[k*(simul.PointsPerElement-1) + 1] = 2*simul.uNow[k*(simul.PointsPerElement-1) + 1] - simul.uPrev[k*(simul.PointsPerElement-1) + 1] -
                                                                    2*simul.timestep^2*wFactor*sum +
                                                                    simul.timestep^2*wFactor*(0.5*simul.QuadWeights[simul.PointsPerElement]*(simul.nodes[k+1] - simul.nodes[k]) + 0.5*simul.QuadWeights[1]*(simul.nodes[k+2] - simul.nodes[k+1]))*(-simul.fVals[(k-1)*(simul.PointsPerElement-1) + 1]*cos(simul.omega*(stepnumber-1)*simul.timestep))

            end
        end     
    
    end

    
    #update the variables
    for i = 1:length(simul.uPrev)
        simul.uPrev[i] = simul.uNow[i]
        simul.uNow[i] = simul.uNext[i]
    end   

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

end