using ANOVAapprox
using Distributions
using Random
using GroupedTransforms
using LinearAlgebra
using Test
using Plots
using IterativeSolvers
using ToeplitzMatrices
using SparseArrays
using StatsBase

using Distributions





function fun(X)
    if size(X) == (1,)
        return X
    elseif size(X,1) == 1
    return  cos.(X[1,:]) .*exp.(X[1,:]) #.* cos.(2*π .* X[1,:])
   end
   if size(X,1) == 2
   return y = (max.(1/9 .- (X[1,:]).^2,0)).* (max.(1/9 .- (X[2,:]).^2,0))
  end
end





d = 1
q = 1
m = 2
MSE1 = Dict()
ns = 2:7
m = 3
for n in ns
    #n = 5

    η = (m-1) * 2^(-Float64(n))
    #η =0

    M = 2^(n+1)*(n+1)^d

    mu = Uniform(eta,1)
    X = rand(mu, d, M)
    y = fun(X)


        
    X_transformed = (η .+ ((1-η) .* cdf.(mu,X))).-0.5
    X_test =  rand(mu,d,3*M)
    y_test = fun(X_test)

    X_test_t = (η .+ ((1-η) .* cdf.(mu,X_test))) .-0.5



    # use rho to build transformation with cummulative distribution function
    if m == 2
        apx = ANOVAapprox.approx( X_transformed, y, d, Int.(n*ones(d)),"chui2" )
    elseif m == 3
        apx = ANOVAapprox.approx( X_transformed, y, d, Int.(n*ones(d)),"chui3" )
    elseif m ==4
        apx = ANOVAapprox.approx( X_transformed, y, d, Int.(n*ones(d)),"chui4" )
    end

    ANOVAapprox.approximate( apx,  lambda= [0.0], max_iter=500)
    MSE1[n] = ANOVAapprox.get_mse(apx,X_test_t,y_test,0.0)

    
end
#plot(MSE1,label="non-per")
plot(MSE1,label="n", yscale = :log10)
plot!(ns,  10 .* MSE1[ns[1]] .* 2 .^ collect(-2.0 .*ns), label="s=1")
plot!(ns, 10 .* MSE1[ns[1]] .* 2 .^ collect(-4.0 .*ns), label="s=2")
plot!(ns, 10 .* MSE1[ns[1]] .* 2 .^ collect(-6.0 .*ns), label="s=3")
