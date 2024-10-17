# ANOVA_RFF: ANOVA-Boosting for random Fourier Features
This algorithms aims to do sensitivity analysis for high-dimensional data, even for dependent input variables. 
Notes for the implementation:


- test_anova_RFF.jl:
Test file for testing the RFF-Boosting.

- anova_RFF.jl:
Creates module anova_rff

- algs_RFF.jl:
Containes all algortihms used for the ANOVA-Boosting. Implemented random feature algorithms to apply after sensitivity analysis are SHRIMP and HARFE.


Notes for the implementation:
The function `f = anova_RFF.RFF_model(X,y, "exp")` constructs a RFF model, where $X$ is the matrix of inputs and $y$ is the vector of outputs. 
The function `U = anova_RFF.ANOVA_boosting(f,q,N)` finds an ANOVA-truncated index set $U$. You have additional Parameters to choose:
- `dependence`: `true` or `false` $\rightarrow$ there are two different possibilities implemented for the dependence of the Input, see paper. The option `true` can also be applied to Independent Input
- `anova_step`: 'ascent' or 'descent' $\rightarrow$ either start with all all terms of order $1$ and increase the order iterativly to $q$ or start with all terms of order $q$ and delete nonimportant terms of highest order iterativly. See corresponding Dissertation for more Information.   
- `epsilon`: threshold parameter
 
Once you have found an index-set $U$, you can apply a RFF algorithm by `anova_RFF.shrimp(X,y, U, N)`, which draws random Features according to the index set $U$.



All algorithms are implemented for the expoential basis $\mathrm e ^{\mathrm{i} \langle w, x \rangle}$, but you can add other functions: in the function `make_A` and in the module `anova_rff`
U = anova_RFF.ANOVA_boosting(shr,q,N, dependence = true, anova_step = anova_step, epsilon = epsilon)



This is the repository for the algorithms described in the paper

ANOVA-Boosting for Random Fourier Features
Daniel Potts, Laura Weidensager
*arXiv: 2404.03050*



In the folder *Dissertation* you can find all numerical Experiments for my Dissertation.




