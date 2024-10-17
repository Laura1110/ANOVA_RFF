using IterativeSolvers
using SpecialFunctions
using LinearAlgebra
using Plots 
using CSV 
using DataFrames
include("anova_RFF.jl")

#### test functions: 
function fun(X,f)   
    if f ==1
        #return X[4,:].^2 ./ (1 .+X[4,:].^2 ).^2 .+  X[3,:] .* X[2,:] ./ ( (1 .+X[2,:].^2 ) .* (1 .+X[3,:].^2 )) .+  X[1,:] .* X[2,:] ./ ( (1 .+X[1,:].^2 ) .* (1 .+X[2,:].^2 )).+  exp.(-X[4,:])      #fT1
        return X[4,:].^2 .+  X[3,:] .* X[2,:] .+  X[1,:] .* X[2,:] .+  X[4,:]      #fT1
    elseif f ==2
        	return sin.(X[1,:]) + 7 .* sin.(X[2,:]).^2 + 0.1.* X[3,:].^4 .*  sin.(X[1,:]) #fT2
    elseif f ==3
        return 10 .* sin.(pi .*  X[1,:] .* X[2,:]) + 20 .* (X[3,:]  .- 0.5).^2 + 10 .* X[4,:]  + 5 .*  X[5,:]   #f3
        
    elseif f == 4
        return X[4,:] .+  X[3,:] .+  X[2,:] .+  X[1,:]    
    elseif f == 5
        return 2\ sqrt(6) .* (- max.(X[1,:],X[2,:]) - max.(X[3,:],X[4,:]) - max.(X[5,:],X[6,:]) .+1)
    end
end

#   number and distribution of sampling points:
M =200
sample_dist = Uniform(0,1) 
#   choose function: 
f = 3

#   choose paramters for approximation:                
d = 5 
q = 2
epsilon = 0.001
anova_step = "ascent" 

#   consrtuct samples:
X = anova_RFF.make_X(d,M,dist = sample_dist)
X_test = anova_RFF.make_X(d,M,dist = sample_dist)
y = Complex.(fun(X,f))
y_test = Complex.(fun(X_test,f))

### ANOVA-Boosting:
N = 5*M
shr = anova_RFF.RFF_model(X,y, "exp")
U = anova_RFF.ANOVA_boosting(shr,q,N, dependence = true, anova_step = anova_step, epsilon = epsilon)
println("found the ANOVA index set U: ", U)
U = anova_RFF.anti_downward_closed(U)
println("anti_dw closed set: ", U)


N = 5*M
shr = anova_RFF.shrimp(X,y, U, N)
mse1 = anova_RFF.get_mse(shr, X_test,y_test)
println("MSE with shrimp:", mse1)

shr2 = anova_RFF.harfe(X,y, U, N)
mse2 = anova_RFF.get_mse(shr2, X_test,y_test)
println("MSE with harfe:", mse2)



