"""
Modelling microprocessor cooling
Anni Kauniskangas, 2020

This script produces results for the different setups. Note: Running each part can take a long time!
"""
import numpy as np
import matplotlib.pyplot as plt
from classes import Microprocessor
from classes import WithCase
from classes import WithHeatSink

# Validation test (microprocessor only) --------------------------------------------------------------------------------
Microchip_test = Microprocessor(0.014,0.001,0.0005,8630,5*10**8)
Microchip_test.gauss_seidel_simple("Plot")

# Microchip and case with natural convection ---------------------------------------------------------------------------
Case_test = WithCase(0.020,0.002,0.012,0.001,0.0005,5500,5*10**8)
Case_test.gauss_seidel_solve("Plot")

# Heatsink, case and chip with natural convection-----------------------------------------------------------------------

#Create a plot of the scenario
Heatsink_test = WithHeatSink(0.046,0.030,0.020,0.002,0.014,0.001,0.001,0.002,0.028,0.001,100,5*10**8)
Heatsink_test.gauss_seidel_solve("Plot")

# Vary the number of fins while keeping fin spacing and width constant
file = open("num_fins.txt","a+")  # Save results to a file for later use

Num_Fins_4 = WithHeatSink(0.020,0.030,0.020,0.002,0.014,0.001,0.002,0.004,0.026,0.001,900,5*10**8)
Num_Fins_8 = WithHeatSink(0.044,0.030,0.020,0.002,0.014,0.001,0.002,0.004,0.026,0.001,500,5*10**8)
Num_Fins_10 = WithHeatSink(0.056,0.030,0.020,0.002,0.014,0.001,0.002,0.004,0.026,0.001,300,5*10**8)
Num_Fins_12 = WithHeatSink(0.068,0.030,0.020,0.002,0.014,0.001,0.002,0.004,0.026,0.001,100,5*10**8)
Temps =[]
Temps.append(Num_Fins_4.gauss_seidel_solve("Value"))
file.write("4," + str(Temps[0])+"\n")
Temps.append(Num_Fins_8.gauss_seidel_solve("Value"))
file.write("8," + str(Temps[1])+"\n")
Temps.append(Num_Fins_10.gauss_seidel_solve("Value"))
file.write("10," + str(Temps[2])+"\n")
Temps.append(Num_Fins_12.gauss_seidel_solve("Value"))
file.write("12," + str(Temps[3])+"\n")
file.close()

x = [4,8,10,12]
plt.plot(x,Temps,"x", markersize="12")
plt.xlabel("Number of fins", fontsize="20")
plt.ylabel("Average Temperature", fontsize="20")
plt.show()

# Heatsink, case and chip with forced convection------------------------------------------------------------------------
Heatsink_Fan = WithHeatSink(0.056,0.030,0.020,0.002,0.014,0.001,0.002,0.004,0.026,0.001,100,5*10**8)
Heatsink_Fan.gauss_seidel_solve(mode="Plot",conv="forced")