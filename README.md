
2D HEAT TRANSFER USING FINITE ELEMENT ANALYSIS

The purpose of the code written here is to numerically predict the outcome of heat transfer within certain structures (as seen below), with the corresponding boundary conditions.

<img width="444" height="248" alt="Screenshot 2026-09-11 at 01 39 33" src="https://github.com/user-attachments/assets/5a80f6cd-5310-4630-b058-64bed4f61976" />




<img width="444" height="248" alt="Screenshot 2026-09-11 at 01 40 19" src="https://github.com/user-attachments/assets/58654d8d-b7b5-4eec-93b9-3fcc77632fe1" />


The constants corresponding to the problem are: k = 1.5 W/m·°C, h = 50 W/m²·°C, T∞ = 25°C.

The notation used is: q = heat flow, k = thermal conductivity, h = heat transfer coefficient, T∞ = ambient temperature.

The Python code given in this repo has the purpose of taking an input file of the structure's characteristics and producing a stiffness matrix K. The stiffness matrix K is assembled by combining the local stiffness matrices of each individual triangular element. After assembling the K matrix, the boundary conditions (Dirichlet, Neumann, Robin) are applied.

Applying the boundary conditions will define a thermal load vector F.

Finally, we solve the equation KU = F, with U being the temperature of each node in the structure.




