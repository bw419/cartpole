"""
fork from python-rl and pybrain for visualization
"""
from globals import *
#import numpy as np
import autograd.numpy as np
from matplotlib.pyplot import ion, draw, Rectangle, Line2D
import matplotlib.pyplot as plt


class CartPole:
	"""Cart Pole environment. This implementation allows multiple poles,
	noisy action, and random starts. It has been checked repeatedly for
	'correctness', specifically the direction of gravity. Some implementations of
	cart pole on the internet have the gravity constant inverted. The way to check is to
	limit the force to be zero, start from a valid random start state and watch how long
	it takes for the pole to fall. If the pole falls almost immediately, you're all set. If it takes
	tens or hundreds of steps then you have gravity inverted. It will tend to still fall because
	of round off errors that cause the oscillations to grow until it eventually falls.
	"""

	def __init__(self, delta_time=.2, visual=False, frictionless=False):
		self.cart_location = 0.0
		self.cart_velocity = 0.0
		self.pole_angle = np.pi    # angle is defined to be zero when the pole is upright, pi when hanging vertically down
		self.pole_velocity = 0.0
		self.visual = visual

		# Setup pole lengths and masses based on scale of each pole
		# (Papers using multi-poles tend to have them either same lengths/masses
		#   or they vary by some scalar from the other poles)
		self.pole_length = 0.5 
		self.pole_mass = 0.5 

		self.frictionless = frictionless
		self.mu_c = 0.001 #   # friction coefficient of the cart
		self.mu_p = 0.001 #   # friction coefficient of the pole

		if self.frictionless:
			self.mu_c = 0
			self.mu_p = 0
		
		self.sim_steps = 50 #50        # number of Euler integration steps to perform in one go
		self.delta_time = delta_time #.2      # time step of the Euler integrator
		self.max_force = 20.
		self.gravity = 9.8
		self.cart_mass = 0.5

		# for plotting
		self.cartwidth = 1.0
		self.cartheight = 0.2
		
		self.dyn_noise1 = 0.0
		self.dyn_noise2 = 0.0

		if self.visual:
			self.drawPlot()

	def setDynamicNoise(self, dyn_noise):
		self.dyn_noise1 = 16.0*dyn_noise
		self.dyn_noise2 = 41.6*dyn_noise


	def setState(self, state):
		self.cart_location = state[0]
		self.cart_velocity = state[1]
		self.pole_angle = state[2]
		self.pole_velocity = state[3]
			
	def getEnergy(self, custom_params=False):
		state = self.getState()
		if custom_params:
			V = 0.5 * self.pole_length * self.gravity * self.pole_mass * (np.cos(state[2]) - 1)
			T = 0.5 * (self.cart_mass + self.pole_mass) * state[1]**2
			T += 0.5 * self.pole_mass * self.pole_length * state[3] * state[1] * np.cos(state[2])
			T += (0.5 * self.pole_mass / 3) * (self.pole_length * state[3])**2
			return [T, V]
		return get_state_energy(state)

	def getPoleEnergy(self, custom_params=False):
		state = self.getState()
		if custom_params:
			V = 0.5 * self.pole_length * self.gravity * self.pole_mass * (np.cos(state[2]) - 1)
			T = (0.5 * self.pole_mass / 3) * (self.pole_length * state[3])**2
			return T + V
		return get_state_energy(state)

	def getState(self, energy=False):
		if not energy:
			return np.array([self.cart_location,self.cart_velocity,self.pole_angle,self.pole_velocity])
		T, V = self.getEnergy()
		return np.array([self.cart_location,self.cart_velocity,self.pole_angle,self.pole_velocity, T, V])

	# reset the state vector to the initial state (down-hanging pole)
	def reset(self):
		self.cart_location = 0.0
		self.cart_velocity = 0.0
		self.pole_angle = np.pi
		self.pole_velocity = 0.0

	# This is where the equations of motion are implemented
	def performAction(self, action = 0.0):
		# prevent the force from being too large

		force = self.max_force * np.tanh(action/self.max_force)

		# integrate forward the equations of motion using the Euler method
		for step in range(50):#self.sim_steps):
			s = np.sin(self.pole_angle)
			c = np.cos(self.pole_angle)
			m = 4.0*(self.cart_mass+self.pole_mass)-3.0*self.pole_mass*(c**2)
			

			cart_accel = (2.0*(self.pole_length*self.pole_mass*(self.pole_velocity**2)*s+2*(force-self.mu_c*self.cart_velocity))\
				-3.0*self.pole_mass*self.gravity*c*s + 6.0*self.mu_p*self.pole_velocity*c/self.pole_length)/m + np.random.normal(scale=self.dyn_noise1)
			
			pole_accel = (-3.0*c*2.0/self.pole_length*(self.pole_length/2.0*self.pole_mass*(self.pole_velocity**2)*s + force-self.mu_c*self.cart_velocity)+\
				6.0*(self.cart_mass+self.pole_mass)/(self.pole_mass*self.pole_length)*\
				(self.pole_mass*self.gravity*s - 2.0/self.pole_length*self.mu_p*self.pole_velocity) \
				)/m + np.random.normal(scale=self.dyn_noise2)


			# Update state variables
			dt = (self.delta_time / float(self.sim_steps))
			# Do the updates in this order, so that we get semi-implicit Euler that is simplectic rather than forward-Euler which is not. 
			self.cart_velocity += dt * cart_accel
			self.pole_velocity += dt * pole_accel
			self.pole_angle    += dt * self.pole_velocity
			self.cart_location += dt * self.cart_velocity

		if self.visual:
			self._render()

	# remapping as a member function
	def remap_angle(self):
		self.pole_angle = _remap_angle(self.pole_angle)
	
   # the following are graphics routines
	def drawPlot(self):
		ion()
		self.fig = plt.figure()
		# draw cart
		self.axes = self.fig.add_subplot(111, aspect='equal')
		self.box = Rectangle(xy=(self.cart_location - self.cartwidth / 2.0, -self.cartheight), 
							 width=self.cartwidth, height=self.cartheight)
		self.axes.add_artist(self.box)
		self.box.set_clip_box(self.axes.bbox)

		# draw pole
		self.pole = Line2D([self.cart_location, self.cart_location + np.sin(self.pole_angle)], 
						   [0, np.cos(self.pole_angle)], linewidth=3, color='black')
		self.axes.add_artist(self.pole)
		self.pole.set_clip_box(self.axes.bbox)

		# set axes limits
		self.axes.set_xlim(-10, 10)
		self.axes.set_ylim(-0.5, 2)
		
	def _render(self):
		self.box.set_x(self.cart_location - self.cartwidth / 2.0)
		self.pole.set_xdata([self.cart_location, self.cart_location + np.sin(self.pole_angle)])
		self.pole.set_ydata([0, np.cos(self.pole_angle)])
		self.fig.show()
		
		plt.pause(0.05)



#################################################
# Useful functions to interact with CartPole.py:


sys = CartPole(0.2, False)

def assert_t_step(t_step):
	global sys
	if sys.delta_time != t_step:
		sys = CartPole(t_step, False)






# New state variables after evolving for 1 timestep
# Returns a 4-vector, given either 4- or 5-vector.

def single_action(state, t_step=0.2):
	if len(state) == 4:
		return single_action4(state)
	else:
		return single_action5(state)


def single_action4(state4, t_step=0.2):

	global sys
	assert_t_step(t_step)

	sys.setState(state4)
	sys.performAction(0.)

	return np.array(sys.getState())


def single_action5(state5, t_step=0.2):

	global sys
	assert_t_step(t_step)

	sys.setState(state5[:4])
	sys.performAction(state5[4])

	return np.array(sys.getState())


def single_action(state4, action, t_step=0.2):

	global sys
	assert_t_step(t_step)

	sys.setState(state4)
	sys.performAction(action)

	return np.array(sys.getState())



# Target function - the change in a set of state variables over the timestep

def target(state, t_step=0.2):
	if len(state) == 4:
		return single_action4(state) - np.array(state)
	else:
		return single_action5(state) - np.array(state[:4])

def target4(state, t_step=0.2):
	return single_action4(state) - np.array(state)

def target5(state, t_step=0.2):
	return single_action5(state) - np.array(state[:4])



# If theta  has gone past our conceptual limits of [-pi,pi]
# map it onto the equivalent angle that is in the accepted range (by adding or subtracting 2pi)
def remap_angle(theta):
	while theta < -np.pi:
		theta += 2. * np.pi
	while theta > np.pi:
		theta -= 2. * np.pi
	return theta
  
remap_angle_v = np.vectorize(remap_angle)

def remapped_angle(state):
	return np.concatenate((state[:2], [remap_angle(state[2])], state[3:]))


# # If theta  has gone past our conceptual limits of [-pi,pi]
# # map it onto the equivalent angle that is in the accepted range (by adding or subtracting 2pi)
# def remap_angle(theta):
# 	abs_theta = np.abs(theta)
# 	sign = np.sign(theta)



# 	while theta < -np.pi:
# 		theta += 2. * np.pi
# 	while theta > np.pi:
# 		theta -= 2. * np.pi
# 	return theta
  
# remap_angle_v = np.vectorize(remap_angle)



# converts eg. target(x) to single_action(x)
def to_update_fn_w_action(fn):
	def new_fn(x, action=0.0):
		return fn([x[0], x[1], x[2], x[3], action]) + x[:4]
	return new_fn

def to_update_fn(fn):
	def new_fn(x):
		return fn(x) + x[:4]
	return new_fn

# converts eg. single_action(x) to target(x)
def to_diff_fn(fn):
	def new_fn(x):
		return fn(x) - x[:4]
	return new_fn



def get_state_energies(state):
	V = 0.125 * 9.8 * (np.cos(state[2]) - 1)
	T = 0.5 * state[1]**2
	T += 0.125 * state[3] * state[1] * np.cos(state[2])
	T += (0.0625 / 3) * (state[3])**2
	return [T, V]

def get_tot_state_energy(state):
	V = 0.125 * 9.8 * (np.cos(state[2]) - 1)
	T = 0.5 * state[1]**2
	T += 0.125 * state[3] * state[1] * np.cos(state[2])
	T += (0.0625 / 3) * (state[3])**2
	return T+V

def get_tot_pole_energy(state):
	return 0.125 * 9.8 * (np.cos(state[2]) - 1) + (0.0625 / 3) * (state[3])**2




import ctypes
import pathlib
from subprocess import Popen, PIPE, run


funcs = []

def intialise_cfuncs():
	global funcs

	fun = ctypes.CDLL("../libfun1.so")

	out = Popen(
	    args="nm ../libfun1.so", 
	    shell=True, 
	    stdout=PIPE
	).communicate()[0].decode("utf-8")

	attrs = [
	    i.split(" ")[-1].replace("\r", "") 
	    for i in out.split("\n") if " T " in i
	]

	func_names = [func_name for func_name in attrs if hasattr(fun, func_name)]
	print(func_names)
	funcs = [getattr(fun, func_name) for func_name in func_names]

	offset = 0

	funcs[0].argtypes = [ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float]
	# funcs[8].argtypes = [ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float]
	for i in [1, 2, 3, 4]:#, 9, 10, 11, 12]:
		funcs[i].restype = ctypes.c_float

	funcs[5].argtypes = [ctypes.c_uint]
	funcs[6].restype = ctypes.c_float
	funcs[7].argtypes = [ctypes.c_float]

	seed = np.random.randint(0, 10000000)
	seed_C_noise(seed)



# def try_kfn(state, action):
# 	funcs[8](ctypes.c_int(256), *[ctypes.c_float(x) for x in [state[0], state[1], state[2], state[3], action]])
# 	return [x() for x in funcs[9:13]]

def seed_C_noise(seed):
	funcs[5](ctypes.c_uint(seed))

def set_C_dynamic_noise(dyn_noise):
	funcs[7](ctypes.c_float(dyn_noise))


def fast_single_action(state, action=0.0):
	funcs[0](*[ctypes.c_float(x) for x in [state[0], state[1], state[2], state[3], action]])
	return [x() for x in funcs[1:5]]


def fast_target(state, t_step=0.2):
	if len(state) == 4:
		return fast_single_action(state, 0.) - np.array(state)
	else:
		return fast_single_action(state[:4], state[4]) - np.array(state[:4])


def single_action_perf_comparison():

	print(single_action([1.,1.,1.,1.],1.))
	print(fast_single_action([1.,1.,1.,1.],1.))

	t = time.perf_counter()
	for i in range(1000):
		single_action([1.,1.,1.,1.],1.)

	time1 = time.perf_counter() - t
	t = time.perf_counter()

	for i in range(1000):
		fast_single_action([1.,1.,1.,1.],1.)

	time2 = time.perf_counter() - t

	print(time1/time2, f"times faster ({time1:.4f}ms, {time2:.4f}ms)")


	# t = time.perf_counter()

	# for i in range(1000):
	# 	try_kfn([1.,1.,1.,1.],1.)

	# time3 = time.perf_counter() - t
	# print(time3)




def set_dynamic_noise(dyn_noise=None):
	if dyn_noise is None:
		dyn_noise = 0.


	sys.setDynamicNoise(dyn_noise)
	set_C_dynamic_noise(dyn_noise)



if __name__ == "__main__":
	print("recompiling c.")
	run(["cc", "-shared", "-o", "../libfun1.so", "../cfuncs.cpp"])

	intialise_cfuncs()
	single_action_perf_comparison()

else:
	intialise_cfuncs()