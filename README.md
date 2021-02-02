# MicroprocessorCooling
This code solves for the steady-state heat transport in a 2D model of a microprocessor, ceramic casing and an aluminium heatsink. It uses either Jacobi or Gauss-Seidel relaxation method on a finite difference grid. It can be run with the microprocessor only, microprocessor and casing, or microprocessor with casing and heatsink. Options for either forced or natural convection. Result can be plotted as a temperature heatmap over the grid.

Anni Kauniskangas
2020

Files included(3)
- This README file
- classes.py
- script.py


Instructions:

In order to run the code, make sure that all the .py files listed above are in the same working directory. The python packages required to run the code are: NumPy, decimal, matplotlib and seaborn. Seaborn is used for creating heat maps of the results, but the code can also be run without it by changing the mode in the gauss_seidel_solve calls from ”Plot” to ”Value”.

To run the code and produce results, it is only required to run script.py However, all of the methods and attributes required for the simulation are contained within the three classes in classes.py.

Please note: Running each part of script.py can take a long time! Uncomment one section at a time to run
