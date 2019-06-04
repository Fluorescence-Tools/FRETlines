# FRET-lines generator

This is a small Python package for the calculation of FRET-lines for use in single-molecue FRET experiments using multiparameter fluorescence detection<sup>[1]</sup>. These experiments are analyzed using two-dimensional histograms of the FRET efficiency, *E*, and the intensity-weighted average donor fluorescence lifetime, <&tau;<sub>D(A)</sub>><sub>F</sub>. FRET-lines define relationships between these two observables for different physical models of the system that can be overlayed on the plot for comparison to the experimental data<sup>[2]</sup>.

For an ideal system, the two observables are related by the *ideal static FRET-line*:

*E* = 1- <&tau;<sub>D(A)</sub>><sub>F</sub>/&tau;<sub>D(0)</sub>
  
where &tau;<sub>D(0)</sub> is the donor only fluorescence lifetime. To model the influence of the flexible dye linkers, static FRET-lines can be generated based on a normal or &chi; distribution for the interdye distance. In addition, two polymer models are implemented: the random coil (or Gaussian chain) and the worm-like chain (WLC).

Dynamic FRET-lines characterize the exchange between distinct conformations and show as a curve connecting two points on the static FRET-line. An example is given below.

![Example of a dynamic FRET-line](/docs/dynamic_FRET_line.png)
 
 # Usage
 
 The usage of the package is demonstrated in the associated [Jupyter Notebook.](https://github.com/AndersBarth/FRETlines/blob/master/FRETlines.ipynb)

# Installation

The FRETlines package can either be run from the root folder, or be installed to your local Python installation by `python setup.py install`. It is also available through PyPI by `pip install FRETlines`.

## Dependencies

numpy, numba

# References

[1] Eggeling C, et al. (2001) Data registration and selective single-molecule analysis using multi-parameter fluorescence detection. Journal of Biotechnology 86(3):163–180.

[2] Kalinin S, Valeri A, Antonik M, Felekyan S, Seidel CAM (2010) Detection of structural dynamics by FRET: a photon distribution and fluorescence lifetime analysis of systems with multiple states. J Phys Chem B 114(23):7983–7995.

