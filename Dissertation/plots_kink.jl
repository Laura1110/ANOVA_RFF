using ANOVAapprox
using GroupedTransforms
using LinearAlgebra
using Plots


d = 3
m = 2
ds = d
#M = 10000
println("d = ", d)
# d=1
# m=2
# ds = d
#n = 20

bw = Int.(9*ones(d))

if d == 1
    N = CWWTtools.datalength(bw)+1
elseif d == 2
    N = CWWTtools.datalength(bw)+2*CWWTtools.datalength([bw[1]])+1
elseif d == 3
    N = CWWTtools.datalength(bw)+3*CWWTtools.datalength([bw[1],bw[2]])+3*CWWTtools.datalength([bw[1]])+1
end

M = 3*Int(ceil(N*log(N))) #logarithmic oversampling


λs = [0.0]

X = rand(  d, M ) .- 0.5

""" function values """
function fun(X)
    if size(X,1) == 1
       # return sin.(n .* pi .* X[1,:])
        return sqrt(3545/16)*(max.(1/9 .- (X[1,:]).^2,0))
    elseif size(X,1) == 2
        return 3545/16 .* (max.(1/9 .- (X[1,:]).^2,0)) .* (max.(1/9 .- (X[2,:]).^2,0))
    elseif size(X,1) == 3
        return 3545/16 * sqrt(3545/16) .* (max.(1/9 .- (X[1,:]).^2,0)) .* (max.(1/9 .- (X[2,:]).^2,0)) .* (max.(1/9 .- (X[3,:]).^2,0))
    end
end
y = fun(X)
a = ANOVAapprox.approx( X, y, d, bw,"chui2")
ANOVAapprox.approximate( a, lambda = [0.0], tol = 10^(-12))
println("Approximation finished")


""" sum of wavelet coefficients """
n_max = maximum(bw)
global sum_wav_coeff = zeros(n_max)

for u in a.U
    if length(u)>0
    freq = CWWTtools.cwwt_index_set(Int.(n_max*ones(length(u))))
        if length(u) ==1
            freq = freq'
        end
        global ac_co = 1
        for i = 1:size(freq,2)
            j = freq[:,i]
            n = sum(j)
            if n >0
                ## supremum:
                global sum_wav_coeff[n] = max(sum_wav_coeff[n],sum(abs.(a.fc[0.0][u][ac_co:ac_co+2^(sum(j))-1]).^2))   
                # sum:
                #global sum_wav_coeff[n] = sum_wav_coeff[n] + sum(abs.(a.fc[0.0][u][ac_co:ac_co+2^(sum(j))-1]).^2)   
            end
                global ac_co = ac_co + 2^sum(j)
        end
    end
end



for i = 1:length(sum_wav_coeff)
              println("(",i,",",sum_wav_coeff[i],")")
 end


#  plot(collect(1:9),collect(2^(4*i)*sum_wav_coeff[i] for i in 1:9))


# collect(2^(4*i)*sum_wav_coeff[i] for i in 1:9)


# sum_wav_coeff[9]/n^(m-1)