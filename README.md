# FRET-lines generator

This is a small Python package for the calculation of FRET-lines for use in single-molecue FRET experiments using multiparameter fluorescence detection. These experiments are analyzed using two-dimensional histograms of the FRET efficiency, *E*, and the intensity-weighted average donor fluorescence lifetime, <&tau;<sub>D(A)</sub>><sub>F</sub>. FRET-lines define relationships between these two observables for different physical models of the system that can be overlayed on the plot for comparison to the experimental data.

For an ideal system, the two observables are related by the *ideal static FRET-line*:

*E* = 1- <&tau;<sub>D(A)</sub>><sub>F</sub>/&tau;<sub>D(0)</sub>
  
 where &tau;<sub>D(0)</sub> is the donor only fluorescence lifetime. To model the influence of the flexible dye linkers, static FRET-lines can be generated based on a normal or &chi; distribution for the interdye distance. In addition, two polymer models are implemented: the random coil (or Gaussian chain) and the worm-like chain (WLC).
 
 # Usage
 
 The usage of the package is demonstrated in the associated [Jupyter Notebook.](https://github.com/AndersBarth/FRETlines/blob/master/FRETlines.ipynb)

# Installation

The FRETlines package can either be run from the root folder, or be installed to your local Python installation by `python setup.py install`. It is also available through PyPI by `pip install FRETlines`.
