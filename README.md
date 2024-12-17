# ANOVA_RFF: ANOVA-Boosting for Random Fourier Features
This algorithms aims to do sensitivity analysis for high-dimensional data, even for dependent input variables. 

## Included files

- test_anova_RFF.jl:<br>
Test file for testing the RFF-Boosting.

- anova_RFF.jl:<br>
Creates module `anova_rff`.

- algs_RFF.jl:<br>
Containes all algortihms used for the ANOVA-Boosting. 


## Notes on the implementation
The function 
```
f = anova_RFF.RFF_model(X,y, "exp")
```
constructs a RFF model, where $X$ is the matrix of inputs and $y$ is the vector of outputs. 
The function 
```
U = anova_RFF.ANOVA_boosting(f,q,N)
```
 finds an ANOVA-truncated index set $U$. There are additional parameters to choose:
- `dependence`: `true` or `false` $\rightarrow$ there are two different possibilities implemented for the dependence of the input, see paper. The option `true` can also be applied to Independent Input
- `anova_step`: 'ascent' or 'descent' $\rightarrow$ either start with all all terms of order $1$ and increase the order iteratively to $q$ or start with all terms of order $q$ and delete nonimportant terms of highest order iteratively. See corresponding Dissertation for more information.   
- `epsilon`: threshold parameter
 
Once you have found an index-set $U$, you can apply a RFF algorithm by e.g. `anova_RFF.shrimp(X,y, U, N)`, which draws random Features according to the index set $U$. Implemented random feature algorithms to apply after sensitivity analysis are SHRIMP and HARFE.



All algorithms are implemented for the expoential basis $\mathrm e ^{\mathrm{i} \langle w, x \rangle}$, but you can add other functions: in the function `make_A` and in the module `anova_rff`.


## References

This is the repository for the algorithms described in the paper

ANOVA-Boosting for Random Fourier Features<br>
Daniel Potts, Laura Weidensager<br>
*ArXiv:* [2404.03050](https://arxiv.org/abs/2404.03050)



In the folder *Dissertation* you can find all numerical experiments for my Dissertation:
Figure 4.1: plots_kink.jl
Figure 6.8-6.11: interpretability.jl

Table 6.1: independent.jl
Table 6.2: numerik_dep.jl
Table 6.4: validate_trafo.jl, validate_RFF.jl












