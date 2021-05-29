from globals import *

# from os import urandom
# seed = urandom(16)

P_RANGE5 = np.array([15, 10, np.pi, 15, 25])
# P_BOUNDS5 = np.ones((4, 2))
# P_BOUNDS5 *= P_RANGE4[:, np.newaxis]
P_RANGE4 = P_RANGE5[:4]
P_RANGE = P_RANGE5

TRAIN_P_RANGE5 = P_RANGE5#np.array([16, 11, np.pi, 16, 26])
TRAIN_P_RANGE4 = TRAIN_P_RANGE5[:4]
TRAIN_P_RANGE = TRAIN_P_RANGE5


VAR_STR = (r"$x$", r"$\dot{x}$", r"$\theta$", r"$\dot{\theta}$", "$F$")



def format_IC(IC):

	IC_strs = []
	for val in IC:
		IC_strs.append(f"{val:.2f}")

	if IC[2] == np.pi:
		IC_strs[2] = r"$\pi$"
	if IC[2] == -np.pi:
		IC_strs[2] = r"$-\pi$"
	if IC[2] == np.pi/2:
		IC_strs[2] = r"$-\pi/2$"
	if IC[2] == -np.pi/2:
		IC_strs[2] = r"$-\pi/2$"


	return "[" + ", ".join(IC_strs) + "]"


############## RANDOM STATE GENERATION ##################
def rand_states(bounds):
	bounds = np.array(bounds)

	state = np.random.random(len(bounds)) * 2 - 1

	return state * bounds

def rand_states4(bounds=None):
	if bounds is None:
		bounds = P_RANGE4
	return rand_state(bounds)


def rand_states5(bounds=None):
	if bounds is None:
		bounds = P_RANGE5
	return rand_state(bounds)


def rand_state(bounds):
	bounds = np.array(bounds)

	state = np.random.random(len(bounds)) * 2 - 1

	return state * bounds

def rand_state4(bounds=None):
	if bounds is None:
		bounds = P_RANGE4
	return rand_state(bounds)


def rand_state5(bounds=None):
	if bounds is None:
		bounds = P_RANGE5
	return rand_state(bounds)


# Best to do this as a power of 2
# because it ensures balance
def sobol_rand_states(N, bounds):
	bounds = np.array(bounds)
	states = sobol_seq.i4_sobol_generate(len(bounds), N)* 2 - 1
	return states * bounds[np.newaxis, :]


def sobol_rand_states4(N, bounds=None):
	if bounds is None:
		bounds = P_RANGE4
	return sobol_rand_states(N, bounds)

def sobol_rand_states5(N, bounds=None):
	if bounds is None:
		bounds = P_RANGE5
	return sobol_rand_states(N, bounds)


sobol_seed = 1
def sobol_init():
	global sobol_seed
	sobol_seed = 1

def sobol_rand_state(bounds):
	global sobol_seed
	bounds = np.array(bounds)
	state, sobol_seed = sobol_seq.i4_sobol(len(bounds), sobol_seed)
	state = state * 2 - 1
	return state * bounds[np.newaxis, :]

def sobol_rand_state4(bounds=None):
	if bounds is None:
		bounds = P_RANGE4
	return sobol_rand_state(bounds)

def sobol_rand_state5(bounds=None):
	if bounds is None:
		bounds = P_RANGE5
	return sobol_rand_state(bounds)

#########################################################


def contour_plot(fn_to_plot, start_state=None, bounds=None, xi=2, yi=3, si=None, NX=50, NY=50, NS=8, ax=None, incl_f=True, pi_multiples=True):

	if ax is None:
		ax = plt.gca()

	if bounds is None:
		if incl_f:
			bounds = P_RANGE5
		else:
			bounds = P_RANGE4

	x = np.linspace(-bounds[xi], bounds[xi], NX)
	y = np.linspace(-bounds[yi], bounds[yi], NY)
	X, Y = np.meshgrid(x, y)
	Z = np.zeros_like(X)

	if start_state is None:
		start_state = np.zeros(len(bounds))

	# if si is not None:
	# for k, s_val in enumerate()

	for i, x_val in enumerate(x):
		for j, y_val in enumerate(y):

			state = start_state
			state[xi] = x_val
			state[yi] = y_val

			Z[j, i] = fn_to_plot(state)
				

	im = ax.contourf(X, Y, Z)
	ax.set_xlabel(VAR_STR[xi])
	ax.set_ylabel(VAR_STR[yi])
	
	divider = make_axes_locatable(ax)
	cax = divider.append_axes("right", size="5%", pad=0.05)
	cb = plt.colorbar(im, cax=cax)

	plt.sca(ax)

	if pi_multiples:
		if xi == 2: axis_pi_multiples(ax.xaxis)
		if yi == 2: axis_pi_multiples(ax.yaxis)



def line_plot(fn_to_plot, start_state=None, bounds=None, xi=2, NX=50, ax=None, incl_f=True):

	if ax is None:
		ax = plt.gca()

	if bounds is None:
		if incl_f:
			bounds = P_RANGE5
		else:
			bounds = P_RANGE4

	x = np.linspace(-bounds[xi], bounds[xi], NX)
	z = np.zeros_like(x)

	if start_state is None:
		start_state = np.zeros(len(bounds))

	for i, x_val in enumerate(x):

		state = start_state
		state[xi] = x_val

		z[i] = fn_to_plot(state)
				

	ax.plot(x, z)
	ax.set_xlabel(VAR_STR[xi])
	
	if xi == 2: axis_pi_multiples(ax.xaxis)


def show_matrix(M, zero_range = 0, title="", x="", y="", axes=True, div=True, cmap_str=None):

	if cmap_str is not None:
		cmap = cm.get_cmap(cmap_str, 256)
		newcolors = cmap(np.linspace(0, 1, 256))
	else:
		if div:
			cmap = cm.get_cmap("RdYlBu", 256)
			newcolors = cmap(np.linspace(1, 0, 256))
		else:
			cmap = cm.get_cmap("viridis", 256)
			newcolors = cmap(np.linspace(0, 1, 256))


	newcolors[:zero_range, :3] = 0
	newcmp = ListedColormap(newcolors)

	if axes:
		plt.gca().set_xticks([0,1,2,3])
		plt.gca().set_xticklabels(VAR_STR)

		plt.gca().set_yticks([0,1,2,3])
		plt.gca().set_yticklabels(VAR_STR)
		plt.xlabel(x, labelpad=2.0)
		plt.ylabel(y)
	else:
		plt.gca().set_xticks([])
		plt.gca().set_yticks([])

	plt.title(title)

	plt.imshow(M, cmap = newcmp)
	plt.colorbar()
	if div:
		maxelem = np.max(np.abs(M[np.isnan(M)==False]))
		plt.clim(-maxelem, maxelem)
	
	print(M)
	
	plt.show()





# https://stackoverflow.com/questions/40642061/how-to-set-axis-ticks-in-multiples-of-pi-python-matplotlib
def multiple_formatter(denominator=2, number=np.pi, latex='\pi'):
	def gcd(a, b):
		while b:
			a, b = b, a%b
		return a

	def _multiple_formatter(x, pos):
		den = denominator
		num = np.int(np.rint(den*x/number))
		com = gcd(num,den)
		(num,den) = (int(num/com),int(den/com))
		if den==1:
			if num==0:
				return r'$0$'
			if num==1:
				return r'$%s$'%latex
			elif num==-1:
				return r'$-%s$'%latex
			else:
				return r'$%s%s$'%(num,latex)
		else:
			if num==1:
				return r'$\frac{%s}{%s}$'%(latex,den)
			elif num==-1:
				return r'$-\frac{%s}{%s}$'%(latex,den)
			else:
				return r'$\frac{%s%s}{%s}$'%(num,latex,den)
	return _multiple_formatter


class Multiple:
	def __init__(self, denominator=2, number=np.pi, latex='\pi'):
		self.denominator = denominator
		self.number = number
		self.latex = latex

	def locator(self):
		return plt.MultipleLocator(self.number / self.denominator)

	def formatter(self):
		return plt.FuncFormatter(multiple_formatter(self.denominator, self.number, self.latex))



def axis_pi_multiples(axis_obj):
	axis_obj.set_major_locator(plt.MultipleLocator(np.pi / 2))
	axis_obj.set_minor_locator(plt.MultipleLocator(np.pi / 12))
	axis_obj.set_major_formatter(plt.FuncFormatter(multiple_formatter()))
