# Elliptic-grid


1. The airfoil coordinates in the code itself for each each case.
2. Download all the three python files. The files are in .py format.
3. Make sure that numpy matplotlib and scipy libraries are installed in the your VS code
4.Open the spline fitting file and run it. Choose run without debugging option and then choose python debugger.
5.Repeat the step 4 for TFI grid and ellptic grid python file.
6. The codes will geenrate the respective plots available in report.
7. Following replacement can be used in line 21 and 22 for both TFI grid and  elliptic grid to generate grid having airfoil with 1m chord.
      
        x = airfoil_data[i] / 100  # Convert mm to meters
        y = airfoil_data[i+1] / 100 # Consodering chord to be 100 cm instead of 100mm and other coordinates of airfoil surfaces in meter.
