# Bilinear bandit e-FALB

This is the code for the paper 'Improved Regret Bounds of Bilinear Bandits using Action Space Dimension Analysis' published in ICML 2021.

Required software
 - python 3 with numpy, scipy, sklearn, cython, ipdb, cvxpy, tqdm

Compile needed
 - in `matrixrecovery`, run `cython myutils_cython.pyx` and then `python3 setup.py install`
 - for mac
    - in `pyOptSpace_py3_custom`, run `cython optspace.pyx` then `python3 setup.py install`
    - copy the created `*.so` file to the upper directory
 - for linux
    - do the same as mac above, but in the directory `pyOptSpace_py3_linux_custom`

To replicate the plot in the paper (experiment data already run)
 - run `python3 analyze_expr01_20210205_paper.py`

To replicate the result in the paper
  1. Run the script ./running_script.sh
  2. Move all .pkl files to a subfolder in /res-20210205/R001/T10000 folder 
  3. remove all the date prefix on .pkl file name. For example, change the name '20210204Thu-232134-bmoracle.pkl' to 'bmoracle.pkl'
  4. Run `python3 analyze_expr01_20210205_paper.py`
  5. If you want to replicate other results in the appendix of our paper, follow the same steps with appropriate changes on T and R variables in 'running_script.sh', `python3 analyze_expr01_20210205_paper.py` (here they are total_time and RRR), and subfolder name. 


<!--
# License

This SDK is distributed under the [Apache License, Version 2.0](http://www.apache.org/licenses/LICENSE-2.0), see [LICENSE](./LICENSE) and [NOTICE](./NOTICE) for more information.
-->
