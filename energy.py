from globals import *
from model import *
from rollout_utils import *
import visvis as vv


def plot_energy():
	sys = CartPole(0.02)
	sys.setState([0, 0, 0.2, 0])

	states = []
	Es = []
	T = [x for x in range(100)]
	for t in T:

		print(sum(sys.getEnergy()))

		sys.performAction(0.)
		states.append(sys.getState())
		Es.append(sys.getEnergy())

	plt.plot(T, [x[0] for x in Es], label="T")
	plt.plot(T, [x[1] for x in Es], label="V")
	plt.plot(T, [sum(x) for x in Es], label="T+V")
	plt.plot(T, [x[0] for x in states], label="x")
	plt.plot(T, [x[2] for x in states], label="theta")
	plt.legend()
	plt.show()



def energy_ellipse_params(E, theta):
	RHS = E + (9.8/8)*(1-np.cos(theta))
	a = 0.5
	b = .0625 * np.cos(theta)
	c = 1/48

	conditions = (a*c - b*b > 0, RHS / (a + c) > 0)
	# print(conditions)
	if not all(conditions):
		return

	sqrt_term = np.sqrt((a-c)**2 + 4 * b**2)
	major = np.sqrt(2*RHS / (a + c - sqrt_term))
	minor = np.sqrt(2*RHS / (a + c + sqrt_term))
	angle = .5*np.pi + .5*np.arctan2(2*b, (a-c))

	return major, minor, angle
	# print(major, minor, 180/np.pi * (angle-0.5*np.pi))

def ellipse_fn(angle, pos, a, b, tilt_angle):
	x_ = a * np.cos(angle)
	z_ = b * np.sin(angle)
	x = x_*np.cos(tilt_angle) - z_*np.sin(tilt_angle)
	y = np.zeros_like(angle) + pos
	z = x_*np.sin(tilt_angle) + z_*np.cos(tilt_angle)

	return x, y, z

def get_IC(E, theta, phi):

	ret = energy_ellipse_params(E, theta)
	if ret is None:
		return None

	major, minor, tilt_angle = ret
	x, y, z = ellipse_fn(phi - tilt_angle, theta, major, minor, tilt_angle)

	return [0, x, y, z]

def plot_energy_ellipse(E, theta):

	major, minor, tilt_angle = energy_ellipse_params(E, theta)


	angle = np.linspace(0, 2*np.pi, 100)
	x,y,z = ellipse_fn(t, theta, major, minor, tilt_angle)
	vv.plot(x, y, z, lc=(0, 0, 1), alpha=0.5, lw=3)


def isosurface_rings(ax, E, theta_range, phi_range=(-np.pi, np.pi), N_thetas=50, N_phis=50):
	# thetas = np.linspace(-5.5*np.pi, 2.5*np.pi, 50)
	thetas = np.linspace(theta_range[0], theta_range[1], N_thetas)
	phis = np.linspace(phi_range[0], phi_range[1], N_phis)

	x_grid = np.zeros(thetas.shape + phis.shape)
	y_grid = np.zeros(thetas.shape + phis.shape)
	z_grid = np.zeros(thetas.shape + phis.shape)
	fc_grid = np.ones(thetas.shape + phis.shape + (4,))

	# Thetas, Phis = np.meshgrid(thetas, phis, sparse=False, indexing='ij')

	params = np.array([np.array(energy_ellipse_params(E, theta)) for theta in thetas])
	# print(params)

	cmap = cm.get_cmap('summer', 256)

	for i, theta in enumerate(thetas):
		major, minor, tilt_angle = params[i]

		x, y, z = ellipse_fn(phis, theta, major, minor, tilt_angle)
		x_grid[i,:] = x 
		y_grid[i,:] = y
		z_grid[i,:] = z
			
		c = cmap(0.5 + 3*(tilt_angle - .5*np.pi))

		fc_grid[i,:,:] = (c[0], c[1], c[2], .99) 
	

	vv.xlabel(r"cart velocity")
	vv.ylabel(r"pole angle")
	vv.zlabel(r"pole angular velocity")
	vv.axis("off")
	vv.surf(x_grid, y_grid*3, z_grid, fc_grid, axes_adjust=True)
	# ax.plot_surface(x_grid, y_grid, z_grid, facecolors=fc_grid, shade=True, rstride=1, cstride=1)

	# for i, theta in enumerate(thetas):
	# 	vv.plot(grid[i,:,0],grid[i,:,1], grid[i,:,2], alpha=1, lw=3)
	# for i, phi in enumerate(phis):
	# 	vv.plot(grid[:,i,0],grid[:,i,1], grid[:,i,2], alpha=1, lw=3)

	# for i, phi in enumerate(phis):
		# major, minor, tilt_angle = params[i]

		# x,y,z = ellipse_fn_2(phi, thetas, major, minor, tilt_angle)
		# vv.plot(x, y, z, lc=(0, 0, 1), alpha=0.5, lw=3)
	






def draw_energy_trajectories(E=1):

	app = vv.use()

	f = vv.clf()
	ax = vv.cla()

	# isosurface_rings(ax, E, (-0.1*np.pi, 2.1*np.pi)) #E>0
	# isosurface_rings(ax, E, (0.01*np.pi, 1.99*np.pi)) #E=0
	# isosurface_rings(ax, E, (0.3*np.pi, 1.7*np.pi)) #E=-0.5
	isosurface_rings(ax, E, ((1-.28195)*np.pi, (1+.28195)*np.pi)) #E=-2
	# isosurface_rings(ax, E, (0.95*np.pi, 1.05*np.pi)) #E=-2.4349
	# isosurface_rings(ax, E, ((1-.0912)*np.pi, (1+.0912)*np.pi)) #E=-2.5

	ICs1 = [get_IC(E, 0, 0.5*np.pi), 
			get_IC(E, 0, 0.3*np.pi),
			get_IC(E, 0, 0.17*np.pi)]


	# ICs2 = [get_IC(E, 1.01*np.pi, 0),
	# 		get_IC(E, 1.05*np.pi, 0),
	# 		get_IC(E, 1.08*np.pi, 0),
	# 		get_IC(E, 1.0911*np.pi, 0)]

	ICs2 = [get_IC(E, 1.1*np.pi, 0),
			get_IC(E, 1.2*np.pi, 0),
			get_IC(E, 1.28195*np.pi, 0)] #E = -2

	# ICs2 = [get_IC(E, 1.95*np.pi, 0), 
	# 		get_IC(E, 1.5*np.pi, 0),
	# 		get_IC(E, 1.25*np.pi, 0)] #E>0

	ICs1 = [x for x in ICs1 if x is not None]
	ICs2 = [x for x in ICs2 if x is not None]


	states1_slow = [rollout(IC, 20, 0.2) for IC in ICs1]
	states1_fast = [rollout(IC, 200, 0.02) for IC in ICs1]
	# states2_slow = [rollout(IC, 20, 0.2) for IC in ICs2]
	# states2_fast = [rollout(IC, 200, 0.02) for IC in ICs2] # E>0
	states2_slow = [rollout(IC, 10, 0.2) for IC in ICs2]
	states2_fast = [rollout(IC, 100, 0.02) for IC in ICs2] #E=-2


	for states_list in [states2_slow, states2_fast, states1_slow, states1_fast]:
		for states in states_list:
			states[:,2][np.abs(states[:,2]) > 2.1*np.pi] = np.nan

	full_plot = True

	sf = 3
	c1 = (138/255,43/255,226/255)
	c2 = "r"
	if full_plot:

		for states in states1_slow:
			vv.plot(states[:,1], states[:,2]*sf, states[:,3], lc=c1, ls="--", alpha=0.99)

		for states in states2_slow:
			vv.plot(states[:,1], states[:,2]*sf, states[:,3], lc=c2, ls="--", alpha=0.99)

		for states in states1_fast:
			vv.plot(states[:,1], states[:,2]*sf, states[:,3], lc=c1, lw=5, alpha=0.99)

		for states in states2_fast:
			vv.plot(states[:,1], states[:,2]*sf, states[:,3], lc=c2, lw=5, alpha=0.99)

		for states in states1_slow:
			vv.plot(states[:,1], states[:,2]*sf, states[:,3], mc=c1, mw=5, mew=3, mec="k", ms="o", ls="", alpha=0.99)

		for states in states2_slow:
			vv.plot(states[:,1], states[:,2]*sf, states[:,3], mc=c2, mw=5, mew=3, mec="k", ms="o", ls="", alpha=0.99)

	else:
		
		for states in states2_fast:
			vv.plot(states[:,1], states[:,2]*sf, states[:,3], lc=c1, lw=5, alpha=0.99)
		
		for states in states1_fast:
			vv.plot(states[:,1], states[:,2]*sf, states[:,3], lc=c2, lw=5, alpha=0.99)


	# isosurface_rings(1)

	app.Run()

plot_energy()
draw_energy_trajectories(E=-2)