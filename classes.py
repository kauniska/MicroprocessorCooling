"""
Modelling microprocessor cooling
Anni Kauniskangas

This module contains the classes Microprocessor, WithCase and WithHeatSink. Microprocessor contains the main methods
for solving the temperature distribution of the grid in 2D, as well as methods for calculating the energy balance and
checking for convergence. The classes WithCase and WithHeatSink inherit from Microprocessor, and are modified to include
a ceramic case or a ceramic case and a heatsink in the main temperature grid.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sb
from decimal import Decimal

class Microprocessor:
    """
    This class contains all the main methods required for solving the heat transport equation on a finite 2-dimensional
    grid. It can only be used to solve for the microprocessor alone, but the main gauss-seidel algorithm is general
    and is used for the classes WithCase and WithHeatSink as well
    """
    def __init__(self,xlength,ylength,spacing,Tguess,power):

        r1 = Decimal(str(xlength)) % Decimal(str(spacing))  # Make sure that the input dimensions will fit an integer
        r2 = Decimal(str(ylength)) % Decimal(str(spacing))  # number of pixels with the given spacing
        if r1.is_zero() and r2.is_zero():
            self._xlength = xlength
            self._ylength = ylength
        else:
            raise Exception("Please make sure that the input dimensions are integer multiples of the spacing!")

        self._power = power  # Power and spacing
        self._h = spacing
        self._v = 20  # Wind speed kept at a constant 20m/s

        self._x = np.arange(self._h/2,self._xlength,self._h)  # x-coordinates of grid points
        self._y = np.arange(self._h/2,self._ylength,self._h)  # y-coordinates of grid points
        self._xpoints = len(self._x)
        self._ypoints = len(self._y)

        # Initialize the temperature grid, including ghost points, based on the initial guess
        #self._T = np.random.random((self._ypoints+2,self._xpoints+2))*10+Tguess  # Random fluctuations
        self._T = np.full((self._ypoints + 2, self._xpoints + 2), Tguess, dtype=float)  # Uniform

        self._k = np.full((self._T.shape),150,dtype=float)  # Initialize the thermal conductivity matrix
        self._up = []  # Initialize list of upper edge points
        self._down = []  # Initialize list of lower edge points
        self._left = []  # Initialize list of left edge points
        self._right = []  # Initialize list of right edge points
        self._empty = []  # Initialize a list of "empty" points
        self._corners_down = []  # Initialize lists of inner corner points
        self._corners_left = []
        self._corners_right = []

        # Now we should have a uniform grid of desired size, with added ghost points for the boundary


    def edges(self):
        """
        Creates the arrays containing all the indices of the surface points in T. This makes it easier to
        perform the calculations for boundary conditions
        :return:
        """
        for j in range(1,self._T.shape[1]-1):  # In this case the edges are just the edges of the grid - ghost points
            self._up.append([1,j])
            self._down.append([-2,j])
        for i in range(1,self._T.shape[0]-1):
            self._left.append([i,1])
            self._right.append([i,-2])


    def empty_points(self):
        """
        Creates a list of the empty points in the grid that should not be updated
        :return:
        """
        self._empty = []  # In the case of only the microprocessor, there are no empty points


    def corners(self):
        """
        Creates a list of inner corners in the grid, so that they can be calculated separately from the main algorithm
        :return:
        """
        self._corners_down = []  # No inner corners for the microprocessor only
        self._corners_left = []
        self._corners_right = []


    def q(self,i,j):
        """
        Returns the power density in a point, given x and y coordinates
        :param x: x-coordinate
        :param y: y-coordinate
        :return: power density q
        """
        if 0 <= j <= self._xpoints and 0 <= i <= self._ypoints:  # In this case the power density is q everywhere
            return self._power
        else:
            return 0


    def k(self):
        """
        Defines the conductivity k everywhere in the grid
        :return:
        """
        self._k *= 1  # In this case k is uniform everywhere


    def area(self):
        A = self._xlength * self._ylength
        return A


    def natural_convection(self):
        """
        Calculates the ghost points outside surfaces, according to the flux out from natural convection. The function
        skips ghost points at the inner corners, as they are surrounded by 2 pixels and have to be dealt with separately
        :return:
        """

        for index in self._up:  # Boundary conditions for the upwards surfaces
            C = -self._h / self._k[index[0],index[1]] * 1.31
            self._T[index[0]-1,index[1]] = C * abs(self._T[index[0],index[1]])**(4/3) + self._T[index[0],index[1]]

        for index in self._down:  # Boundary conditions for the downwards surfaces
            C = -self._h / self._k[index[0], index[1]] * 1.31
            if index in self._corners_down:
                continue
            self._T[index[0]+1,index[1]] = C * abs(self._T[index[0], index[1]]) ** (4/3) + self._T[index[0], index[1]]

        for index in self._left:  # Boundary conditions for the left surfaces
            C = -self._h / self._k[index[0], index[1]] * 1.31
            if index in self._corners_left:
                continue
            self._T[index[0],index[1]-1] = C * abs(self._T[index[0], index[1]]) ** (4/3) + self._T[index[0],index[1]]

        for index in self._right:  # Boundary conditions for the right surfaces
            C = -self._h / self._k[index[0], index[1]] * 1.31
            if index in self._corners_right:
                continue
            self._T[index[0],index[1]+1] = C * abs(self._T[index[0], index[1]]) ** (4/3) + self._T[index[0],index[1]]


    def forced_convection(self):
        """
        Calculates the ghost points outside surfaces, according to the flux out from forced convection. The function
        skips ghost points at the inner corners, as they are surrounded by 2 pixels and have to be dealt with separately
        :return:
        """
        v = self._v  # Wind speed

        for index in self._up:  # Boundary conditions for the upwards surfaces
            C = -self._h / self._k[index[0],index[1]] * (11.4 + 5.7*v)
            self._T[index[0]-1,index[1]] = C * abs(self._T[index[0],index[1]]) + self._T[index[0],index[1]]

        for index in self._down:  # Boundary conditions for the downwards surfaces
            C = -self._h / self._k[index[0], index[1]] * (11.4 + 5.7*v)
            if index in self._corners_down:
                continue
            self._T[index[0]+1,index[1]] = C * abs(self._T[index[0], index[1]]) + self._T[index[0], index[1]]

        for index in self._left:  # Boundary conditions for the left surfaces
            C = -self._h / self._k[index[0], index[1]] * (11.4 + 5.7*v)
            if index in self._corners_left:
                continue
            self._T[index[0],index[1]-1] = C * abs(self._T[index[0], index[1]]) + self._T[index[0],index[1]]

        for index in self._right:  # Boundary conditions for the right surfaces
            C = -self._h / self._k[index[0], index[1]] * (11.4 + 5.7*v)
            if index in self._corners_right:
                continue
            self._T[index[0],index[1]+1] = C * abs(self._T[index[0], index[1]]) + self._T[index[0],index[1]]



    def energy_conservation(self):
        """
        Calculates the difference in the input and output powers of the system to check if energy is conserved.
        Assumes a uniform power density in the microprocessor, and natural convection at the boundaries
        :return: difference in input and output power of the system
        """
        P_in = self._power * self.area()  # Total power in = power density * area
        P_out = 0

        C = self._h * 1.31
        for index in self._up:  # Dissipation from the upwards surfaces
            P_out += C * abs(self._T[index[0], index[1]]) ** (4/3)

        for index in self._down:  # Dissipation from the downwards surfaces
            P_out += C * abs(self._T[index[0], index[1]]) ** (4/3)

        for index in self._left:  # Dissipation from the left surfaces
            P_out += C * abs(self._T[index[0], index[1]]) ** (4/3)

        for index in self._right:  # Dissipation from the right surfaces
            P_out += C * abs(self._T[index[0], index[1]]) ** (4/3)

        return P_in - P_out


    def check_convergence(self,T_previous,T_current):
        """
        Calculates the average change in temperature of two subsequent iterations. The average is taken over the whole
        grid
        :param T_previous: Previous iteration grid values (ndarray)
        :param T_current: Current iteration grid values (ndarray)
        :return: True/False, depending if the change between iterations was small enough
        """
        dT = abs(T_previous - T_current)
        mean_dT = np.average(dT)
        mean_T = np.average(np.absolute(T_previous))
        if mean_dT / mean_T < 0.0000001:
            return True
        else:
            return False


    def jacobi_solve(self, mode, conv="natural"):
        """
        Iterates through the grid, updating every value according to the finite difference stencil and saves updated
        values onto a new matrix. Skips through empty grid points, and checks for inner corner points to calculate those
        separately if needed.
        :param mode: Choose "Plot" to create a heatmap or "Value" to return the average temperature
        :param conv: Mode of convection, "natural" or "forced"
        :return:
        """
        self.edges()  # Create lists containing edge indices
        self.empty_points()  # Create a list of empty points if there are any
        self.k()  # Create a matrix containing the information about conductivity at each point

        # Run the algorithm until the energy is conserved sufficiently and convergion criteria is met
        # or until maximum number of iterations is reached (to stop it running too long)
        for k in range(50000):
            self.natural_convection()  # Calculate ghost points outside surfaces
            T_new = self._T.copy()

            for i in range(1, self._T.shape[0] - 1):  # Iterate over the grid with the Jacobi stencil
                for j in range(1, self._T.shape[1] - 1):
                    # Check for any points needing separate treatment. Check for the most probable outcome, (i.e
                    # a normal point inside grid) first to speed up code

                    if ([i,j] not in self._empty) and ([i,j] not in self._corners_right) and ([i,j] not in
                          self._corners_left) and ([i,j] not in self._corners_down):  # Normal point in grid

                        k1 = 2 / (1 / self._k[i,j-1] + 1 / self._k[i, j])  # Calculate conductances for each direction
                        k2 = 2 / (1 / self._k[i,j+1] + 1 / self._k[i, j])  # from the current cell
                        k3 = 2 / (1 / self._k[i-1,j] + 1 / self._k[i, j])
                        k4 = 2 / (1 / self._k[i+1,j] + 1 / self._k[i, j])

                    elif [i, j] in self._empty:  # Skip empty cells
                        continue

                    elif [i, j] in self._corners_down:  # Downwards facing inner corner surface
                        k1 = 2 / (1 / self._k[i,j-1] + 1 / self._k[i,j])
                        k2 = 2 / (1 / self._k[i,j+1] + 1 / self._k[i,j])
                        k3 = 2 / (1 / self._k[i-1,j] + 1 / self._k[i,j])
                        k4 = self._k[i, j]
                        self._T[i+1,j] = -self._h / k4 * 1.31 * abs(self._T[i,j]) ** (4/3) + self._T[i,j]

                    elif [i, j] in self._corners_right:  # Right facing inner corner surface
                        k1 = 2 / (1 / self._k[i,j-1] + 1 / self._k[i,j])
                        k2 = k[i,j]
                        k3 = 2 / (1 / self._k[i-1,j] + 1 / self._k[i,j])
                        k4 = self._k[i,j]
                        self._T[i,j+1] = -self._h / k2 * 1.31 * abs(self._T[i,j]) ** (4/3) + self._T[i,j]

                    elif [i,j] in self._corners_left:  # Left facing inner corner surface
                        k1 = [i,j]
                        k2 = 2 / (1 / self._k[i,j+1] + 1 / self._k[i,j])
                        k3 = 2 / (1 / self._k[i-1,j] + 1 / self._k[i,j])
                        k4 = self._k[i,j]
                        self._T[i,j-1] = -self._h / k1 * 1.31 * abs(self._T[i,j]) ** (4/3) + self._T[i,j]

                    else:
                        raise Exception("Problem with the gauss-seidel algorithm. Point not in grid")

                    C = (self._h ** 2) * (self.q(i,j))
                    k_sum = k1 + k2 + k3 + k4

                    self._T[i,j] = ((k1 * self._T[i, j-1] + k2 * self._T[i,j+1] + k3 *self._T[i-1,j] +\
                                     k4 * self._T[i+1,j]) + C) / k_sum

            # Check convergence
            if self.energy_conservation() / (self._power * self.area()) < 0.0005 \
                        and self.check_convergence(self._T, T_new):
                break

            self._T = T_new.copy()

        for item in self._empty:
            self._T[item[0], item[1]] = 0

        if mode == "Plot":
            T_plot = np.delete(self._T, (0, -1), axis=0)  # Delete ghost points
            T_plot = np.delete(T_plot, (0, -1), axis=1)
            T_plot += 20  # Add ambient temperature

            mask = np.zeros(self._T.shape)  # Create a mask over empty points
            for item in self._empty:
                mask[item[0], item[1]] = 1
            mask = np.delete(mask, (0,-1), axis=0)
            mask = np.delete(mask, (0,-1),axis=1)

            x = np.round(self._x*1000,3)
            y = np.round(self._y*1000,3)
            fig, ax1 = plt.subplots()
            sb.set(font_scale=1.8)
            ax2 = sb.heatmap(T_plot, cmap="coolwarm", xticklabels=x, yticklabels=y, ax=ax1, mask=mask)
            ax1.tick_params(labelsize=24)
            ax2.figure.axes[-1].set_ylabel("Temperature (°C)", size=28)
            plt.xlabel("x(mm)", fontsize="24")
            plt.ylabel("y(mm)", fontsize="24")
            plt.show()

        if mode == "Value":
            T_average = np.average(self._T[self._T != 0])
            print("The average temperature of the system is "+ str(T_average))
            print("The energy difference P_in - P_out is "+str(self.energy_conservation))
            return T_average


    def gauss_seidel_solve(self, mode, conv="natural"):
        """
        Iterates through the grid, updating every value according to the finite difference stencil as it goes.
        Skips through empty grid points, and checks for inner corner points to calculate those separately if needed
        :param mode: Choose "Plot" to create a heatmap or "Value" to return the average temperature
        :param conv: Mode of convection, "natural" or "forced"
        :return:
        """
        self.edges()  # Create lists containing edge indices
        self.empty_points()  # Create a list of empty points, if there are any
        self.k()  # Create a matrix containing the information about conductivity at each point
        self.corners()  # Create lists of inner corner points

        # Run the algorithm until the energy is conserved sufficiently and convergence criteria is met
        # or until maximum number of iterations is reached (to stop it running too long)
        for k in range(100000):
            T_previous = self._T.copy()
            if conv=="natural":
                self.natural_convection()  # Calculate ghost points outside surfaces
            elif conv=="forced":
                self.forced_convection()

            for i in range(1, self._T.shape[0] - 1):  # Iterate over the grid
                for j in range(1, self._T.shape[1] - 1):
                    # Check for any points needing separate treatment. Check for the most probable outcome, (i.e
                    # a normal point inside grid) first to speed up code

                    if ([i,j] not in self._empty) and ([i,j] not in self._corners_right) and ([i,j] not in
                          self._corners_left) and ([i,j] not in self._corners_down):  # Normal point in grid

                        k1 = 2 / (1 / self._k[i,j-1] + 1 / self._k[i,j])  # Calculate conductances for each direction
                        k2 = 2 / (1 / self._k[i,j+1] + 1 / self._k[i,j])  # from the current cell
                        k3 = 2 / (1 / self._k[i-1,j] + 1 / self._k[i,j])
                        k4 = 2 / (1 / self._k[i+1,j] + 1 / self._k[i,j])

                    elif [i,j] in self._empty:  # Skip empty cells
                        continue

                    elif [i,j] in self._corners_down:  # Downwards facing inner corner surface
                        k1 = 2 / (1 / self._k[i,j-1] + 1 / self._k[i,j])
                        k2 = 2 / (1 / self._k[i,j+1] + 1 / self._k[i,j])
                        k3 = 2 / (1 / self._k[i-1,j] + 1 / self._k[i,j])
                        k4 = self._k[i,j]
                        if conv=="natural":
                            self._T[i+1,j] = -self._h / k4 * 1.31 * abs(self._T[i,j]) ** (4/3) + self._T[i,j]
                        elif conv=="forced":
                            self._T[i+1,j] = -self._h / k4 * (11.4 + 5.7*self._v)* abs(self._T[i,j]) + self._T[i,j]

                    elif [i,j] in self._corners_right:  # Right facing inner corner surface
                        k1 = 2 / (1 / self._k[i,j-1] + 1 / self._k[i,j])
                        k2 = self._k[i,j]
                        k3 = 2 / (1 / self._k[i-1,j] + 1 / self._k[i,j])
                        k4 = self._k[i,j]
                        if conv=="natural":
                            self._T[i,j+1] = -self._h / k2 * 1.31 * abs(self._T[i, j]) ** (4/3) + self._T[i,j]
                        elif conv == "forced":
                            self._T[i,j+1] = -self._h / k2 * (11.4 + 5.7 * self._v)*abs(self._T[i,j]) + self._T[i,j]

                    elif [i,j] in self._corners_left:  # Left facing inner corner surface
                        k1 = self._k[i,j]
                        k2 = 2 / (1 / self._k[i,j+1] + 1 / self._k[i,j])
                        k3 = 2 / (1 / self._k[i-1,j] + 1 / self._k[i,j])
                        k4 = self._k[i,j]
                        if conv=="natural":
                            self._T[i,j-1] = -self._h / k1 * 1.31 * abs(self._T[i,j]) ** (4/3) + self._T[i,j]
                        elif conv=="forced":
                            self._T[i,j-1] = -self._h / k1 * (11.4 + 5.7 * self._v) * abs(self._T[i,j]) + self._T[i,j]

                    else:
                        raise Exception("Problem with the gauss-seidel algorithm. Point not found in grid")

                    C = (self._h ** 2) * (self.q(i,j))
                    k_sum = k1+k2+k3+k4

                    self._T[i,j] = ((k1 * self._T[i,j-1] + k2 * self._T[i,j+1] + k3 *
                                    self._T[i-1,j] + k4 * self._T[i+1,j]) + C) / k_sum

            # Check convergence
            if self.energy_conservation()/(self._power*self.area()) < 0.0005 \
                    and self.check_convergence(T_previous, self._T):
                break

        for item in self._empty:
            self._T[item[0], item[1]] = 0

        if mode == "Plot":
            T_plot = np.delete(self._T, (0, -1), axis=0)  # Delete ghost points
            T_plot = np.delete(T_plot, (0, -1), axis=1)
            T_plot += 20  # Add ambient temperature

            mask = np.zeros(self._T.shape)  # Create a mask over empty points
            for item in self._empty:
                mask[item[0], item[1]] = 1
            mask = np.delete(mask, (0,-1), axis=0)
            mask = np.delete(mask, (0,-1),axis=1)

            x = np.round(self._x * 1000, 3)
            y = np.round(self._y * 1000, 3)
            x = x[1::2]
            y = y[1::2]
            fig,ax1 = plt.subplots()
            fig.subplots_adjust(bottom=0.2)
            sb.set(font_scale=1.7)
            ax2 = sb.heatmap(T_plot, cmap="coolwarm", xticklabels=x, yticklabels=y,ax=ax1,mask=mask)
            ax1.tick_params(labelsize=18)
            ax2.figure.axes[-1].set_ylabel("Temperature (°C)",size=28)
            plt.xlabel("x(mm)",fontsize="24")
            plt.ylabel("y(mm)",fontsize="24")
            plt.show()

        if mode == "Value":
            T_average = np.average(self._T[self._T != 0])
            print("The average temperature of the system is "+ str(T_average))
            print("The energy difference P_in - P_out is " + str(self.energy_conservation))
            return T_average


    def gauss_seidel_simple(self, mode):
        """
        Simplified version that only works for a constant k over the grid and doesn't consider empty or corner points
        :param mode: Determines whether results will be plotted or the average temperature returned
        :return: Average temperature, if mode="Value"
        """
        self.edges()  # Create lists containing edge indices
        self.k()  # Create a matrix containing the information about conductivity at each point

        # Run the algorithm until the energy is conserved sufficiently and convergion criteria is met
        # or until maximum number of iterations is reached (to stop it running too long)
        for k in range(700000):
            T_previous = self._T.copy()
            self.natural_convection()  # Calculate ghost points outside surfaces

            for i in range(1, self._T.shape[0] - 1):  # Iterate over the grid with the Jacobi stencil
                for j in range(1, self._T.shape[1] - 1):
                    if [i,j] in self._empty:
                        continue
                    self._T[i,j] = 1 / 4 * (self._T[i-1,j] + self._T[i+1,j] + self._T[i,j-1] + self._T[i,j+1]) + \
                                    self._h ** 2 * self.q(i,j) / (4 * self._k[i,j])

            # Check convergence
            if self.energy_conservation() / (self._power * self.area()) < 0.000005 \
                    and self.check_convergence(T_previous, self._T):
                break

        if mode == "Plot":
            T_plot = np.delete(self._T, (0, -1), axis=0)
            T_plot = np.delete(T_plot, (0, -1), axis=1)
            T_plot += 20

            x = np.round(self._x * 1000, 3)
            y = np.round(self._y * 1000, 3)
            fig, ax1 = plt.subplots()
            fig.subplots_adjust(bottom=0.2)
            sb.set(font_scale=1.7)
            ax2 = sb.heatmap(T_plot, cmap="coolwarm", xticklabels=x, yticklabels=y, ax=ax1)
            ax1.tick_params(labelsize=20)
            ax2.figure.axes[-1].set_ylabel("Temperature (°C)", size=28)
            plt.xlabel("x(mm)", fontsize="24")
            plt.ylabel("y(mm)", fontsize="24")
            plt.show()

        if mode == "Value":
            T_average = np.average(self._T[self._T != 0])
            print("The average temperature of the system is " + str(T_average))
            print("The energy difference P_in - P_out is " + str(self.energy_conservation))
            return T_average


class WithCase(Microprocessor):
    """
    This class inherits the methods from Microprocessor but is modified to include a rectangular ceramic case on top
    of the microprocessor
    """
    def __init__(self, x_case, y_case, x_mp, y_mp, spacing, Tguess, power):
        Microprocessor.__init__(self, x_mp, y_mp, spacing, Tguess, power)

        r1 = Decimal(str(x_case)) % Decimal(str(spacing))  # Make sure that the input dimensions will fit an integer
        r2 = Decimal(str(y_case)) % Decimal(str(spacing))  # number of pixels with the given spacing
        if r1.is_zero() and r2.is_zero():
            self._xcase = x_case  # Case and microprocessor dimensions
            self._ycase = y_case
            self._xmp = x_mp
            self._ymp = y_mp
        else:
            raise Exception("Please make sure that the input dimensions are integer multiples of the spacing!")

        if x_case >= x_mp:  # x-coordinates of grid points based on the wider element
            self._x = np.arange(self._h/2, x_case, self._h)
            self._xlength = x_case
        else:
            raise Exception("Please enter a case width that is larger than or equal to the microprocessor width. " +
                            "Smaller case is not supported currently")

        self._y = np.arange(self._h/2, y_case+y_mp, self._h)  # y-coordinates of grid points
        self._ylength = y_case + y_mp

        self._xpoints = len(self._x)  # Array lengths
        self._ypoints = len(self._y)
        self._ypoints_case = int(self._ycase/self._h)
        self._xpoints_case = int(self._xcase/self._h)
        self._ypoints_mp = int(self._ymp/self._h)
        self._xpoints_mp = int(self._xmp/self._h)

        # Initialize the temperature grid, including ghost points, based on the initial guess
        # self._T = np.random.random((self._ypoints+2,self._xpoints+2))*10+Tguess  # Random fluctuations
        self._T = np.full((self._ypoints + 2, self._xpoints + 2), Tguess, dtype=float)  # Uniform

        self._k = np.zeros((self._T.shape))  # Thermal conductivity matrix


    def area(self):
        """
        Calculates the total surface area of the system
        :return: Surface area in m^2
        """
        A = self._xcase * self._ycase + self._xmp * self._ymp
        return A


    def k(self):
        """
        Defines the thermal conductivity everywhere in the grid
        :return:
        """
        for i in range(self._k.shape[0]):
            for j in range(self._k.shape[1]):
                if i <= self._ypoints_case + 1:
                    self._k[i,j] = 230  # Case
                else:
                    self._k[i, j] = 150  # Microchip

    def q(self,i,j):
        """
        Returns the power density in a point, given grid indices
        :param i: index i in the grid
        :param j: index j in the grid
        :return: power density q
        """
        gap_width = (self._xpoints - self._xpoints_mp) / 2
        if gap_width < j <= self._xpoints-gap_width and self._ypoints-self._ypoints_mp < i <= self._ypoints:
            return self._power
        else:
            return 0


    def edges(self):
        """
        Creates the arrays containing all the indices of the surface points in T. This makes it easier to
        perform the calculations for boundary conditions
        :return:
        """

        for j in range(1, self._T.shape[1] - 1):  # Top edge of the case
            self._up.append([1, j])

        for j in range(1,self._T.shape[1]-1):  # Downwards pointing edges of case and microprocessor
            if j <= (self._xpoints-self._xpoints_mp)/2 or j > (self._xpoints-self._xpoints_mp)/2+self._xpoints_mp:
                self._down.append([self._ypoints_case,j])
            else:
                self._down.append([-2,j])

        for i in range(1, self._T.shape[0] - 1):  # Left and right edges of the microprocessor and case
            if i <= self._ypoints_case:
                self._left.append([i,1])
                self._right.append([i,-2])
            else:
                self._left.append([i, int((self._xpoints-self._xpoints_mp)/2+1)])
                self._right.append([i, int((self._xpoints-self._xpoints_mp)/2+self._xpoints_mp)])


    def empty_points(self):
        """
        Creates a list of the empty points in the grid that should not be updated
        :return:
        """

        for i in range(self._ypoints_case+1,self._ypoints_case+self._ypoints_mp+1):
            for j in range(1,int((self._xpoints-self._xpoints_mp)/2+1)):
                self._empty.append([i,j])
            for j in range(int((self._xpoints-self._xpoints_mp)/2+self._xpoints_mp+1), self._xpoints+1):
                self._empty.append([i,j])


    def corners(self):
        """
        Creates lists of inner corners in the grid, so that they can be calculated separately from the main algorithm.
        For case+microprocessor, there are 4 inner corner points in total that need special treatment. They are saved
        into separate lists depending on the orientation of the surface
        :return:
        """
        self._corners_down.append([self._ypoints_case,int((self._xpoints-self._xpoints_mp)/2)])
        self._corners_down.append([self._ypoints_case, int((self._xpoints - self._xpoints_mp)/2+self._xpoints_mp+1)])

        self._corners_left.append([self._ypoints_case+1,int((self._xpoints-self._xpoints_mp)/2+1)])
        self._corners_right.append([self._ypoints_case+1, int((self._xpoints - self._xpoints_mp)/2+self._xpoints_mp)])


class WithHeatSink(WithCase):
    """
    This class inherits the methods from Microprocessor but is modified to include a rectangular ceramic case on top
    of the microprocessor, as well as a heatsink on top of the case
    """

    def __init__(self, x_hs, y_hs, x_case, y_case, x_mp, y_mp, fin_width, gap_width, fin_depth, spacing, Tguess, power):
        WithCase.__init__(self, x_case, y_case, x_mp, y_mp, spacing, Tguess, power)

        if fin_depth >= y_hs:
            raise Exception("Fins cannot be taller than the heatsink!")

        r1 = Decimal(str(x_hs)) % Decimal(str(spacing))  # Make sure that the input dimensions will fit an integer
        r2 = Decimal(str(y_hs)) % Decimal(str(spacing))  # number of pixels with the given spacing
        r3 = Decimal(str(fin_width)) % Decimal(str(spacing))
        r4 = Decimal(str(gap_width)) % Decimal(str(spacing))
        if r1.is_zero() and r2.is_zero() and r3.is_zero() and r4.is_zero():
            self._xhs = x_hs  # Heatsink parameters
            self._yhs = y_hs
            self._fin_width = fin_width
            self._gap_width = gap_width
        else:
            raise Exception("Please make sure that the input dimensions are integer multiples of the spacing!")

        self._ypoints_hs = int(self._yhs / self._h)

        self._y = np.arange(self._h / 2, y_hs + y_case + y_mp, self._h)  # y-coordinates of grid points
        self._ylength = y_hs + y_case + y_mp
        self._ypoints = len(self._y)

        if x_hs >= x_case:  # x-coordinates of grid points based on the wider element
            self._x = np.arange(self._h / 2, x_hs, self._h)
            self._xlength = x_hs
            self._xpoints = len(self._x)
        else:
            raise Exception("Please enter a heatsink width that is larger than or equal to the case width. " +
                            "Smaller heatsink is not supported currently")

        self._fin_depth = fin_depth
        self._xpoints_fin = int(self._fin_width/self._h)
        self._xpoints_gap = int(self._gap_width/self._h)
        self._num_fins = (self._xpoints + self._xpoints_gap) / ((self._fin_width + self._gap_width) / self._h)
        print("The number of fins is "+str(self._num_fins))

        # Make sure that the created heatsink will be symmetric, with n fins and n-1 gaps
        if self._num_fins * self._xpoints_fin + ((self._num_fins-1) * self._xpoints_gap) != self._xpoints:
            raise Exception("The width of the heatsink cannot fit the fins symmetrically with the current fin " +
                            "parameters. Please change the heatsink width or the fin parameters to crete a symmetric "+
                            "heatsink")

        # Initialize the temperature grid, including ghost points, based on the initial guess
        # self._T = np.random.random((self._ypoints+2,self._xpoints+2))*10+Tguess  # Random fluctuations
        self._T = np.full((self._ypoints + 2, self._xpoints + 2), Tguess, dtype=float)  # Uniform

        self._k = np.zeros(self._T.shape)  # Initialize the thermal conductivity matrix


    def area(self):
        """
        Calculates the total surface area of the system
        :return: Surface area in m^2
        """
        A_hs = self._num_fins * self._fin_width * self._fin_depth + self._xhs * (self._yhs-self._fin_depth)
        A_case = self._xcase * self._ycase
        A_mp = self._xmp * self._ymp
        A = A_hs + A_case + A_mp
        return A

    def k(self):
        """
        Defines the thermal conductivity everywhere in the grid
        :return:
        """
        for i in range(self._k.shape[0]):
            for j in range(self._k.shape[1]):
                if i <= self._ypoints_hs:
                    self._k[i,j] = 250  # Heatsink
                elif i <= self._ypoints_hs + self._ypoints_case:
                    self._k[i,j] = 230  # Case
                else:
                    self._k[i,j] = 150  # Microchip


    def edges(self):
        """
        Creates the arrays containing all the indices of the surface points in T. This makes it easier to
        perform the calculations for boundary conditions
        :return:
        """
        n = 1
        while n <= self._xpoints-self._xpoints_fin + 1:  # Top edges of the heatsink fins
            for j in range(self._xpoints_fin):
                self._up.append([1,n+j])
            n += self._xpoints_fin + self._xpoints_gap

        n = self._xpoints_fin + 1
        while n <= self._xpoints-self._xpoints_fin:  # Upwards edges of the heatsink gaps
            for j in range(self._xpoints_gap):
                self._up.append([int(self._fin_depth/self._h) + 1, n+j])
            n += self._xpoints_fin + self._xpoints_gap

        for j in range(1, self._T.shape[1] - 1):  # Downwards pointing edges of the heatsink, case and microprocessor
            if j <= (self._xpoints - self._xpoints_case) / 2 or j > (self._xpoints - self._xpoints_case) / 2 \
                    + self._xpoints_case:
                self._down.append([self._ypoints_hs, j])
            elif j <= (self._xpoints - self._xpoints_mp) / 2 or (self._xpoints - self._xpoints_mp) / 2 \
                    + self._xpoints_mp < j <= self._xpoints - (self._xpoints - self._xpoints_case) / 2:
                self._down.append([self._ypoints_hs+self._ypoints_case, j])
            else:
                self._down.append([-2, j])

        for i in range(1, self._T.shape[0] - 1):  # Left and right edges of the heatsink, microprocessor and case
            if i <= self._ypoints_hs:
                self._left.append([i, 1])
                self._right.append([i, -2])
            elif self._ypoints_hs < i <= self._ypoints_hs + self._ypoints_case:
                self._left.append([i,int((self._xpoints - self._xpoints_case)/2+1)])
                self._right.append([i,int((self._xpoints - self._xpoints_case)/2 + self._xpoints_case)])
            else:
                self._left.append([i, int((self._xpoints - self._xpoints_mp) / 2 + 1)])
                self._right.append([i, int((self._xpoints - self._xpoints_mp) / 2 + self._xpoints_mp)])

        j = 1
        while j <= self._xpoints - (self._xpoints_fin - 1):  # Left and right edges of the heatsink fins
            for i in range(1, int((self._fin_depth/self._h)+1)):
                if j != 1:  # Prevent saving the long left edge again
                    self._left.append([i,j])
                if j + (self._xpoints_fin-1) != self._xpoints:  # Prevent saving the long right edge again
                    self._right.append([i,j + (self._xpoints_fin-1)])
            j += self._xpoints_fin + self._xpoints_gap


    def empty_points(self):
        """
        Creates a list of the empty points in the grid that should not be updated
        :return:
        """
        m = self._xpoints_fin + 1
        while m <= self._xpoints - self._xpoints_fin:  # Gaps between heatsink fins
            for j in range(self._xpoints_gap):
                for i in range(1, int((self._fin_depth / self._h) + 1)):
                    self._empty.append([i,j+m])
            m += self._xpoints_fin + self._xpoints_gap

        for i in range(self._ypoints_hs + 1,self._ypoints_hs+self._ypoints_case+1):  # Empty points on the case sides
            for j in range(1,int((self._xpoints-self._xpoints_case)/2+1)):
                self._empty.append([i,j])
            for j in range(int((self._xpoints-self._xpoints_case)/2+self._xpoints_case+1), self._xpoints+1):
                self._empty.append([i,j])

        for i in range(self._ypoints_hs + self._ypoints_case +1, self._ypoints+1):  # Empty points on the mp sides
            for j in range(1,int((self._xpoints-self._xpoints_mp)/2)+1):
                self._empty.append([i,j])
            for j in range(int((self._xpoints-self._xpoints_mp)/2+self._xpoints_mp+1),self._xpoints +1):
                self._empty.append([i,j])


    def corners(self):
        """
        Creates lists of inner corners in the grid, so that they can be calculated separately from the main algorithm.
        For case+microprocessor+heatsink, there are 8 main inner corner points in total that need special treatment.
        The inner corners between each fin are not considered here.
        :return:
        """
        self._corners_down.append([self._ypoints_hs, int((self._xpoints-self._xpoints_case)/2)])
        self._corners_down.append([self._ypoints_hs, int((self._xpoints-self._xpoints_case)/2 + self._xpoints_case+1)])

        self._corners_down.append([self._ypoints_hs+self._ypoints_case, int((self._xpoints - self._xpoints_mp) / 2)])
        self._corners_down.append([self._ypoints_hs+self._ypoints_case, int((self._xpoints - self._xpoints_case)/2 \
                                   +self._xpoints_case)])

        self._corners_left.append([self._ypoints_hs + 1, int((self._xpoints - self._xpoints_case) / 2 + 1)])
        self._corners_right.append([self._ypoints_hs + 1, int((self._xpoints - self._xpoints_case) / 2 + self._xpoints_case)])

        self._corners_left.append([self._ypoints_hs+self._ypoints_case + 1, int((self._xpoints - self._xpoints_mp) / 2 + 1)])
        self._corners_right.append([self._ypoints_hs+self._ypoints_case + 1, int((self._xpoints - self._xpoints_mp) / 2 + self._xpoints_mp)])


