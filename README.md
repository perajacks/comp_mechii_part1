
2D HEAT TRANSFER USING FINITE ELEMENT ANALYSIS

The purpose of the code righten is to numerically predict the outcome of heat transfer within certain structures (as seen bellow) with the corresponding boundary conditions.

<img width="444" height="248" alt="Screenshot 2026-09-11 at 01 39 33" src="https://github.com/user-attachments/assets/5a80f6cd-5310-4630-b058-64bed4f61976" />




<img width="444" height="248" alt="Screenshot 2026-09-11 at 01 40 19" src="https://github.com/user-attachments/assets/58654d8d-b7b5-4eec-93b9-3fcc77632fe1" />


The constants corresponding to the problems are k = 1.5 W/m °C, h = 50 W/m² °C, T∞ = 25°C.

The argo used is: q = heat flow, k = heat conduction, h = heat transfer coefficient ,T∞ = ambient temperature.

The Python code given in this repo has the perpuse of geting an inpute file of the stracture charakteristics and prudusing a stifness matrix K. Stifness matrix K is made by combining the local matrixes of each indivdual triangular element stifnes matrix. After establishing the K matrix we have to apply the boundary condtitions( Diricklet, Newman, Robin)






