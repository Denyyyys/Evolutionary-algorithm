
import matplotlib.pyplot as plt
import numpy as np
from typing import Union, Callable
from functions import DOMAIN



linspace_num = 500
CONTOUR_RADIUS = 5
title_f1 = r'$f_{1}(x_{1}, y_{1})=(x_{1}^{2}+y_{1}-11)^{2}+(x_{1} + y_{1}^{2}-7)^{2}$'
title_f2 = r'$f_{2}(x_{2}, y_{2})=2*x_{2}^{2}+1.05*x_{2}^{4}+x_{2}^{6}/6+x_{2}y_{2} + y_{2}^{2}$'

def plot_surface(
		get_f: Callable[[Union[float, np.ndarray], Union[float, np.ndarray]], Union[float, np.ndarray]],
		f_title):
	"""
	makes a surface plot for specified function
	"""
	x = np.linspace(DOMAIN.x_lower_bound, DOMAIN.x_upper_bound, linspace_num)
	y = np.linspace(DOMAIN.y_lower_bound, DOMAIN.y_upper_bound, linspace_num)
	X, Y = np.meshgrid(x, y)
	Z = get_f(X, Y)
	axes = plt.axes(projection="3d")
	surface = axes.plot_surface(X, Y, Z, cmap="plasma")
	axes.set_xlabel('X-axis')
	axes.set_ylabel('Y-axis')
	axes.set_title(f_title)
	plt.colorbar(surface)

def plot_contour(
		get_f: Callable[[Union[float, np.ndarray], Union[float, np.ndarray]], Union[float, np.ndarray]],
		f_title
):
	"""
	Makes a contour plot for specified function
	"""
	x = np.linspace(DOMAIN.x_lower_bound, DOMAIN.x_upper_bound, linspace_num)
	y = np.linspace(DOMAIN.y_lower_bound, DOMAIN.y_upper_bound, linspace_num)
	X, Y = np.meshgrid(x, y)
	Z = get_f(X, Y)
	# plt.contour(X, Y, Z, CONTOUR_RADIUS)
	# contour = plt.contour(X, Y, Z, levels=10, colors='k')
	# plt.clabel(contour, inline=1, fontsize=10)
	contour = plt.contourf(X, Y, Z, levels=10, cmap='viridis')
	plt.colorbar(contour)
	plt.xlabel("x")
	plt.ylabel("y")
	plt.title(f_title)



def plot_contour_with_two_populations(
		get_f: Callable[[Union[float, np.ndarray], Union[float, np.ndarray]], Union[float, np.ndarray]],
		f_title,
		first_pop_x,
		first_pop_y,
		second_pop_x,
		second_pop_y,
		third_pop_x,
		third_pop_y,
):
	"""
	Makes a contour plot for specified function
	"""
	x = np.linspace(DOMAIN.x_lower_bound, DOMAIN.x_upper_bound, linspace_num)
	y = np.linspace(DOMAIN.y_lower_bound, DOMAIN.y_upper_bound, linspace_num)
	X, Y = np.meshgrid(x, y)
	Z = get_f(X, Y)
	# plt.contour(X, Y, Z, CONTOUR_RADIUS)
	# contour = plt.contour(X, Y, Z, levels=10, colors='k')
	# plt.clabel(contour, inline=1, fontsize=10)
	contour = plt.contourf(X, Y, Z, levels=10, cmap='viridis')
	plt.colorbar(contour)
	plt.scatter(first_pop_x, first_pop_y, c='red', marker='o', label='Points')
	plt.scatter(second_pop_x, second_pop_y, c='green', marker='o', label='Points')
	plt.scatter(third_pop_x, third_pop_y, c='purple', marker='o', label='Points')
	plt.xlabel("x")
	plt.ylabel("y")
	plt.title(f_title)


