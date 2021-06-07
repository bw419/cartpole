from globals import *
from cartpole import *

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





if __name__ == "__main__":

	N = 5000
	deltas = np.zeros((4,N))
	for i in range(N):
		state = rand_state5()

		set_dynamic_noise(0)
		state1 = single_action5(state)

		set_dynamic_noise(1)
		state2 = single_action5(state)

		deltas[:,i] = state1 - state2

	print(np.std(deltas, axis=1))



	deltas = np.zeros((4,N))
	for i in range(N):
		state = rand_state5()

		set_dynamic_noise(0)
		state1 = fast_single_action(state[:4], state[4])

		set_dynamic_noise(1)
		state2 = fast_single_action(state[:4], state[4])

		deltas[:,i] = np.array(state1) - state2

	print(np.std(deltas, axis=1))









#########################################################


def variable_scan(idx, N, bound=None, start=None):

	if bound is None:
		bound = P_RANGE5[idx]

	try:
		bound[0]
		min_bound = bound[0]
		max_bound = bound[1]
	except:
		min_bound = -bound
		max_bound = bound


	if start is None:
		start = rand_state5()
	else:
		start = np.array(start)

	state = start.copy()
	for i, scan_val in enumerate(np.linspace(min_bound, max_bound, N)):
		state[idx] = scan_val
		yield i, state

def variable_scan_2d(idx1, idx2, N1, N2, bound1=None, bound2=None, start=None):

	if bound1 is None:
		bound1 = P_RANGE5[idx]

	if bound2 is None:
		bound2 = P_RANGE5[idx]

	try:
		bound1[0]
		min_bound1 = bound1[0]
		max_bound1 = bound1[1]
	except:
		min_bound1 = -bound1
		max_bound1 = bound1

	try:
		bound1[0]
		min_bound2 = bound2[0]
		max_bound2 = bound2[1]
	except:
		min_bound2 = -bound2
		max_bound2 = bound2


	if start is None:
		start = rand_state5()
	else:
		start = np.array(start)

	state = start.copy()
	for i, scan_val1 in enumerate(np.linspace(min_bound1, max_bound1, N1)):
		state[idx1] = scan_val1
		for j, scan_val2 in enumerate(np.linspace(min_bound2, max_bound2, N2)):
			state[idx2] = scan_val2
			yield i, j, state



def function_scan(fn, idx, N, bound=None, start=None):
	temp = [tup for tup in variable_scan(idx, N, bound, start)]
	return [state[idx] for i, state in x], [fn(state) for i, state in x]




##########################################################


# can be changed to use variable_scan_2d
def contour_plot(fn_to_plot, start_state=None, bounds=None, xi=2, yi=3, si=None, NX=50, NY=50, NS=8, ax=None, cb=True, incl_f=True, pi_multiples=True, levels=None, cmap="viridis", filled=True):

	if ax is None:
		ax = plt.gca()

	if bounds is None:
		if incl_f:
			bounds = P_RANGE5
		else:
			bounds = P_RANGE4

	try:
		bounds[0][0]
		min_bounds = bounds[0]
		max_bounds = bounds[1]
	except:
		min_bounds = -np.array(bounds)
		max_bounds = bounds

	# print("bounds =", min_bounds, max_bounds)

	x = np.linspace(min_bounds[xi], max_bounds[xi], NX)
	y = np.linspace(min_bounds[yi], max_bounds[yi], NY)
	X, Y = np.meshgrid(x, y)
	Z = np.zeros_like(X)

	if start_state is None:
		start_state = np.zeros(len(bounds))
	else:
		start_state = start_state.copy()

	# if si is not None:
	# for k, s_val in enumerate()

	for i, x_val in enumerate(x):
		for j, y_val in enumerate(y):

			state = start_state
			state[xi] = x_val
			state[yi] = y_val

			Z[j, i] = fn_to_plot(state)
				

	if levels is None:
		if filled:
			im = ax.contourf(X, Y, Z, cmap=cmap)
		else:
			im = ax.contour(X, Y, Z, cmap=cmap)
	else:
		if filled:
			im = ax.contourf(X, Y, Z, levels=levels, cmap=cmap)
		else:
			im = ax.contour(X, Y, Z, levels=levels, cmap=cmap)

	ax.set_xlabel(VAR_STR[xi])
	ax.set_ylabel(VAR_STR[yi])

	if cb:
		divider = make_axes_locatable(ax)
		cax = divider.append_axes("right", size="5%", pad=0.05)
		cb = plt.colorbar(im, cax=cax)	
		plt.sca(ax)

	if pi_multiples:
		if xi == 2: axis_pi_multiples(ax.xaxis)
		if yi == 2: axis_pi_multiples(ax.yaxis)

	return X, Y, Z, ax, cb 



def line_plot(fn_to_plot, start_state=None, bounds=None, xi=2, NX=50, ax=None, incl_f=True):

	N_X_cmpts = 5 if incl_f else 4

	if start_state is None:
		start_state = np.zeros(N_X_cmpts)

	if ax is None:
		ax = plt.gca()


	x = np.zeros(NX)
	z = np.zeros(NX)

	for i, state in variable_scan(xi, NX, bounds, start_state):
		x[i] = state[xi]
		z[i] = fn_to_plot(state)
				

	ax.plot(x, z)
	ax.set_xlabel(VAR_STR[xi])
	
	if xi == 2: axis_pi_multiples(ax.xaxis)




def plot_scan_matrix(fn_to_plot, N=50, start_state=None, axs_in=None, noise=None, **kwargs):

	if start_state is None:
		start_state = rand_state5()

	if axs_in is None:
		fig, axs = plt.subplots(4, 5)
	else:
		axs = axs_in


	for i in range(5):
		x = np.zeros(N)
		y = np.zeros((N, 4))
		for j, state in variable_scan(i, N, start=start_state):
			x[j] = state[i]
			y[j,:] = fn_to_plot(state)

		for j in range(4):
			ax = axs[j][i]

			if "fmt" in kwargs:
				kwargs1 = kwargs.copy()
				del kwargs1["fmt"]
				ax.plot(x, y[:,j], kwargs["fmt"], **kwargs1)
			else:
				ax.plot(x, y[:,j], **kwargs)
			

			if axs_in is None:
				ax.set_ylim([-20, 20])

				if j == 3:
					ax.set_xlabel(VAR_STR[i])
				else:
					ax.tick_params(bottom=False, which="both", labelbottom=False)

				if i == 0:
					ax.set_ylabel(VAR_STR[j])
				else:
					ax.tick_params(left=False, which="both", labelleft=False)
				
				if i == 2: 
					axis_pi_multiples(ax.xaxis, minor=False)

	return axs




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



def six_planes(fn, varying_bounds=False, cb=True, axs=None, **kwargs):

	# this order ensures axis overdots are in good positions:
	comb_order = ((0, 1), (0, 2), (1, 2), (1, 3), (3, 0), (3, 2))
	if axs is None:
		fig, axs = plt.subplots(2, 3)
		axs = axs.flatten()

	for i, (xi, yi) in enumerate(comb_order):
		print(f"{xi}, {yi} ({i})\r", end="")
		im=contour_plot(fn, xi=xi, yi=yi, ax=axs[i], cb=False, **kwargs)
		if i%3 or varying_bounds:
			axs[i].tick_params(left=False, which="both", labelleft=False)
		if i<3 or varying_bounds:
			axs[i].tick_params(bottom=False, which="both", labelbottom=False)
		axs[i].scatter(kwargs["start_state"][xi], kwargs["start_state"][yi], c="r", marker="x", zorder=10)

	if cb:
		cmap = kwargs["cmap"] if "cmap" in kwargs else "viridis"
		mappable = plt.cm.ScalarMappable(norm=Normalize(kwargs["levels"][0], kwargs["levels"][-1]), cmap=cmap)
		plt.colorbar(mappable, ax=axs)

	return axs




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



def axis_pi_multiples(axis_obj, minor=True):
	if minor:
		axis_obj.set_major_locator(plt.MultipleLocator(np.pi / 2))
		axis_obj.set_minor_locator(plt.MultipleLocator(np.pi / 12))
	else:
		axis_obj.set_major_locator(plt.MultipleLocator(np.pi))

	axis_obj.set_major_formatter(plt.FuncFormatter(multiple_formatter()))




import dill
def save_model_function(fn, fname, log=True):
	if log: print("saving model...")
	with open("../saved/" + fname, "wb") as file:
		s = dill.dump(fn, file)
	if log: print("saved.")

def load_model_function(fname, log=True):
	if log: print("loading model...")
	with open("../saved/" + fname, "rb") as file:
		f = dill.load(file)
		if log: print("loaded.")
	return f

def save_data(data, fname, log=False):
	if log: print("saving data...")
	with open("../saved/" + fname, "wb") as file:
		s = dill.dump(data, file)
	if log: print("saved.")

def load_data(fname, log=False):
	if log: print("loading data...")
	with open("../saved/" + fname, "rb") as file:
		data = dill.load(file)
		if log: print("loaded.")
	return data

