
import matplotlib.pyplot as plt
import numpy as np
from typing import Union, Callable
from collections import namedtuple



Domain = namedtuple('Domain', ['x_lower_bound', 'x_upper_bound', 'y_lower_bound', 'y_upper_bound'])


DOMAIN = Domain(-5, 5, -5, 5)
CONTOUR_RADIUS = 5
linspace_num = 500
title_f1 = r'$f_{1}(x_{1}, y_{1})=(x_{1}^{2}+y_{1}-11)^{2}+(x_{1} + y_{1}^{2}-7)^{2}$'
title_f2 = r'$f_{2}(x_{2}, y_{2})=2*x_{2}^{2}+1.05*x_{2}^{4}+x_{2}^{6}/6+x_{2}y_{2} + y_{2}^{2}$'

def get_f1(x1: Union[float, np.ndarray], y1: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
	"""
	Get the value of function f1 in point x1, y1
	"""
	return (x1**2 + y1 - 11)**2 + (x1 + y1**2 - 7)**2


def get_f2(x2: Union[float, np.ndarray], y2: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
	"""
	Get the value of function f1 in point x1, y1
	"""
	return 2*(x2**2) + 1.05*(x2**4) + (x2**6)/6 + x2*y2 + y2**2

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
	plt.contour(X, Y, Z, CONTOUR_RADIUS)
	plt.xlabel("x")
	plt.ylabel("y")
	plt.title(f_title)