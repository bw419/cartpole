from globals import *
from utils import *
from linear_model import *
from cartpole import *


SQRT2 = np.sqrt(2)
INV_SQRT2 = 1/np.sqrt(2)

# Scalar version of kernel function
kfn_s4 = lambda x, y, s: np.exp(-(
	np.square((x[0]-y[0])*INV_SQRT2/s[0]) +
	np.square((x[1]-y[1])*INV_SQRT2/s[1]) +
	np.square(np.sin(.5*(x[2]-y[2]))*INV_SQRT2/s[2]) +
	np.square((x[3]-y[3])*INV_SQRT2/s[3])
))
kfn_s5 = lambda x, y, s: np.exp(-(
	np.square((x[0]-y[0])*INV_SQRT2/s[0]) +
	np.square((x[1]-y[1])*INV_SQRT2/s[1]) +
	np.square(np.sin(.5*(x[2]-y[2]))*INV_SQRT2/s[2]) +
	np.square((x[3]-y[3])*INV_SQRT2/s[3]) +
	np.square((x[4]-y[4])*INV_SQRT2/s[4])
))

# Vector version of kernel function
kfn4 = lambda x, y, s: np.exp(-(
	np.square((x[0]-y[:,0])*INV_SQRT2/s[0]) +
	np.square((x[1]-y[:,1])*INV_SQRT2/s[1]) +
	np.square(np.sin(.5*(x[2]-y[:,2]))*INV_SQRT2/s[2]) +
	np.square((x[3]-y[:,3])*INV_SQRT2/s[3])
))

kfn5 = lambda x, y, s: np.exp(-(
	np.square((x[0]-y[:,0])*INV_SQRT2/s[0]) +
	np.square((x[1]-y[:,1])*INV_SQRT2/s[1]) +
	np.square(np.sin(.5*(x[2]-y[:,2]))*INV_SQRT2/s[2]) +
	np.square((x[3]-y[:,3])*INV_SQRT2/s[3]) +
	np.square((x[4]-y[:,4])*INV_SQRT2/s[4])
))

# kfn1 = lambda x, y, s: np.exp(-(
# 	np.square((x[:,0]-y[:,0])*INV_SQRT2/s[0]) +
# 	np.square((x[:,1]-y[:,1])*INV_SQRT2/s[1]) +
# 	np.square(np.sin(.5*(x[:,2]-y[:,2]))*INV_SQRT2/s[2]) +
# 	np.square((x[:,3]-y[:,3])*INV_SQRT2/s[3])
# ))




def nonlinear_training_data(N=512, t_step=0.2, sobol=True, get_saved=True, save=False, fname="lin_train", incl_f=True):

	X, Y, C = linear_fit(N, t_step, sobol, get_saved, save, fname, return_data=True, incl_f=incl_f)

	# err = (Y.T - C @ X.T).T
	err = (Y - X @ C.T)

	# print(Y)
	# print(err)
	# print(np.linalg.norm(Y, "fro"))
	# print(np.linalg.norm(err, "fro"))

	return X, err, C


def nonlinear_fit(X, Y, reg, scales, M=256, t_step=0.2):

	N = len(X)
	if N < M:
		raise Exception(f"N={N} but M={M}!")

	λ = reg
	σ = scales

	K_MM = np.zeros((M, M))
	K_MN = np.zeros((M, N))

	if X.shape[1] == 4:
		kfn_s = kfn_s4
		kfn = kfn4
	else:
		kfn_s = kfn_s5
		kfn = kfn5

	for i in range(M):
		a = kfn(X[i,:], X[i+1:,:], σ)
		K_MN[i,i+1:] = a
	
	K_MN[:,:M] += K_MN[:,:M].T
	K_MN[:,:M] += np.identity(M)
	K_MM = K_MN[:,:M]

	b = K_MN @ Y
	A = K_MN @ K_MN.T + λ * K_MM

	αt, res, rank, s = np.linalg.lstsq(A, b, rcond=None)
	α = αt.T

	def fit_fn(x):
		return (α @ kfn(x, X[:M,:], σ))

	return fit_fn


def fit_errors(X, Y, fit_fn, M):
	Y_est = [fit_fn(x) for x in X[:M]]
	errors = np.sum(np.abs(Y[:M] - Y_est), axis=1)
	return np.std(errors)


def data_errors(X, Y, fit_fn):
	return np.std(np.sum(np.abs(Y - [fit_fn(x) for x in X]), axis=1))


GOOD_PARAMS = [1000, 1000, 0.32, 4.2, 21]

def get_nonlinear_fit(N, M, ret_full_fit=True, return_all=False, incl_f=True):
	print("fitting...")
	
	X, Y, C = nonlinear_training_data(N, sobol=True, get_saved=False, incl_f=incl_f)

	perm = np.random.permutation(np.arange(N))
	centre_idxs = perm[:M]
	other_idxs = perm[M:]
	X = X[perm, :]
	Y = Y[perm, :]

	REG = 1e-4
	good_params = GOOD_PARAMS

	residual_fit = nonlinear_fit(X, Y, REG, good_params, M)

	print("fitting complete.")

	# return residual_fit
	if ret_full_fit:
		def full_fit(x):
			return residual_fit(x) + C @ x

		if return_all:
			return full_fit, X, Y, C
		else:
			return full_fit

	else:
		if return_all:
			return residual_fit, X, Y, C
		else:
			return residual_fit, C


def get_good_nonlinear_fit(speed_tradeoff=True, ret_full_fit=True, return_all=False, incl_f=True):
	
	if speed_tradeoff:
		N = 2**11
		M = 2**9
	else:
		N = 2**13
		M = 2**10

	return get_nonlinear_fit(N, M, ret_full_fit, return_all, incl_f)




def nonlinear_model_slice():

	# X, Y = linear_training_data(256, sobol=True)
	# print(X)

	# fig1 = plt.figure()
	# plt.tricontourf(X[:,2],X[:,3], Y[:,3])
	# plt.scatter(X[:,2], X[:,3], c="k", marker="x", s=10, linewidth=0.5)
	# plt.colorbar()
	# plt.show()

	bounds = P_RANGE.copy()
	bounds[2] = 2*bounds[2]
	fn1, X, Y, C = get_good_nonlinear_fit(ret_full_fit=False, return_all=True)

	# line_plot(lambda x: (target(x))[2], bounds=bounds, NX=100, xi=2)
	# line_plot(lambda x: (modified_lin(C, x))[2], bounds=bounds, NX=100, xi=2)
	# line_plot(lambda x: (target(x) - modified_lin(C, x))[2], bounds=bounds, NX=100, xi=2)
	# plt.show()
	# exit()

	fig2 = plt.figure()
	contour_plot(lambda x: fn1(x)[3], bounds=bounds, NX=100)
	plt.scatter(X[:,2], X[:,3], c="k", marker="x", s=10, linewidth=0.5)



	fig3 = plt.figure()
	contour_plot(lambda x: (target(x) - C @ x)[3], bounds=bounds, NX=100)
	plt.show()

# nonlinear_model_slice()


# fn1, X, Y, C = get_good_nonlinear_fit(ret_full_fit=False, return_all=True)





def optimise_nonlinear_fit(X, Y, test_X, test_Y, scan_vars, scan_ranges, fixed_vars, plot=True, log_scale=True, fixed_colours=True, show_fixed=True, show_min=True, show_time=False, override_label=None):

	# scan_var = 0-4: Component of state vector
	# scan_var = 5: regularisation
	# scan_var = 6: M

	colours = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink"]
	var_strs = VAR_STR + (r"$\lambda$", "$M$")

	# with elements changes later
	optima = fixed_vars.copy()


	for scan_n, s_idx in enumerate(scan_vars):

		if override_label is None:
			label = var_strs[s_idx]
		else:
			label = override_label

		print("scanning", label)

		scan_range = scan_ranges[scan_n]

		fixed_idx = np.searchsorted(scan_range, fixed_vars[s_idx], side="left")
		curr_vars = fixed_vars.copy()

		y = np.zeros_like(scan_range)
		y1 = np.zeros_like(scan_range)
		y2 = np.zeros_like(scan_range)
		for i, s in enumerate(scan_range):

			curr_vars[s_idx] = s

			t0 = time.perf_counter()

			# print(s)
			fit_fn = nonlinear_fit(X, Y, curr_vars[5], curr_vars[:5], int(curr_vars[6]))

			t1 = time.perf_counter()

			train = fit_errors(X, Y, fit_fn, int(curr_vars[6]))
			test = data_errors(test_X, test_Y, fit_fn)
		
			y[i] = test
			y1[i] = train
			y2[i] = t1-t0


		min_idx = np.argmin(y)

		optima[s_idx] = scan_range[min_idx]
		optimal_score = y[min_idx]

		if plot:
			if fixed_colours:
				plt.plot(scan_range, y, c=colours[s_idx], label=label)
				plt.plot(scan_range, y1, "--", lw=0.5, c=colours[s_idx])#, label=f"train, {s_idx}")
				if show_fixed: 
					plt.scatter(scan_range[fixed_idx], y[fixed_idx], marker="o", c=colours[s_idx])
				if show_min: 
					plt.scatter(scan_range[min_idx], y[min_idx], marker="x", c=colours[s_idx])
				if show_time:
					plt.plot(scan_range, y2, lw=0.5, c=colours[s_idx])
			else:
				p = plt.plot(scan_range, y, label=label)
				plt.plot(scan_range, y1, "--", lw=0.5, c=p[0].get_color())#, label=f"train, {s_idx}")
				if show_fixed: 
					plt.scatter(scan_range[fixed_idx], y[fixed_idx], marker="o", c=p[0].get_color())
				if show_min: 
					plt.scatter(scan_range[min_idx], y[min_idx], marker="x", c=p[0].get_color())
				if show_time:
					plt.plot(scan_range, y2, lw=0.5, c=p[0].get_color())

			if log_scale:
				plt.semilogx()
			plt.legend()

	return np.array(optima), optimal_score




def evaluate_nonlinear_fit_score(X, Y, test_X, test_Y, params):

	t0 = time.perf_counter()
	fit_fn = nonlinear_fit(X, Y, params[5], params[:5], int(params[6]))
	t1 = time.perf_counter()

	train = fit_errors(X, Y, fit_fn, int(params[6]))
	test = data_errors(test_X, test_Y, fit_fn)
	
	return test, train, t1






def parameter_optimisation():

	# SCALES = [500, 100, 0.3, 20, 20]
	X, Y, C = nonlinear_training_data(2**9, sobol=True, incl_f=True)
	print(C)
	SCALES = np.std(X, axis=0)
	params = list(SCALES) + [1e-4, 1024]
	SCAN_N = 20

	test_X, test_Y = linear_training_data(2**12, sobol=False, incl_f=True)
	test_Y -= test_X @ C.T


	# plt.figure()
	# optima = optimise_nonlinear_fit([5], [np.power(10, np.linspace(-6, -1, 50))], params, True, True)
	# plt.show()



	params = list(GOOD_PARAMS) + [1e-4, 1024]


	plt.figure()
	for exp in [6,7,8,9,10,11]: # best number is 10, takes a long time from 11. M should be as high as possible
		X, Y, C = nonlinear_training_data(int(2**exp), sobol=True, get_saved=False, incl_f=True)
		optimise_nonlinear_fit(X, Y, test_X, test_Y, [6], [np.power(2, np.linspace(4, min(11,exp-0.1), 10))], params, True, True, False, show_fixed=False, show_min=False, show_time=True, override_label=int(2**exp))
	plt.show()


	plt.figure()
	N_values = np.arange(10.1, 16)
	y_values = np.zeros((len(N_values), 3))
	for i, N in enumerate(N_values):
		print("N =", N)
		X, Y, C = nonlinear_training_data(int(2**N), sobol=True, get_saved=False, incl_f=True)
		y_values[i,:] = evaluate_nonlinear_fit_score(X, Y, test_X, test_Y, params)
		print(y_values[i,:])
	plt.plot(N_values, y_values[:,:2])
	ax2=plt.gca().twinx()
	ax2.plot(N_values, y_values[:,2])
	plt.show()






	X, Y, C = nonlinear_training_data(2**9, sobol=True, get_saved=False, incl_f=True)
	params = list(SCALES) + [1e-4, 2**8]


	test_X, test_Y = linear_training_data(2**12, sobol=False, incl_f=True)
	test_Y -= test_X @ C.T


	plt.figure()
	scan_range = np.power(10, np.linspace(-2, 3, SCAN_N))
	scan_ranges = [scan_range for i in range(5)]
	optima, s = optimise_nonlinear_fit(X, Y, test_X, test_Y, range(5), scan_ranges, params, True, True)
	plt.show()

	print("optima:", optima, "scrore", s)


	for delta in [1, .5, .5, .25, .25, .1]:
		plt.figure()

		prev_log_opt = np.log10(optima)
		scan_ranges = [np.power(10, np.linspace(prev_log_opt[i]-delta, prev_log_opt[i]+delta, SCAN_N)) for i in range(2, 5)]


		optima, s = optimise_nonlinear_fit(X, Y, test_X, test_Y, range(2, 5), scan_ranges, optima, True, True)
		print("optima:", optima, "scrore", s)
	
	plt.show()




	print(optima)
	params = optima






if __name__ == "__main__":

	parameter_optimisation()


	nonlin_fn = get_good_nonlinear_fit(ret_full_fit=True)
	actual_fn = target5


	while True:
		fig, ax = plt.subplots(4, 5)

		start_state = rand_state5()
		for i in range(4):
			f1 = lambda x : actual_fn(x)[i]
			f2 = lambda x : nonlin_fn(x)[i]
			for j in range(5):
				line_plot(f1, start_state, xi=j, ax=ax[i][j])
				line_plot(f2, start_state, xi=j, ax=ax[i][j])
				ax[i][j].set_ylim([-P_RANGE[i], 2*P_RANGE[i]])

		plt.show()




	# saving of models and comparison of execution speeds

	# a= [load_model_function("nonlin_13_11"),
	#  load_model_function("nonlin_16_12"),
	# load_model_function("nonlin_15_13"),
	# single_action5a,
	# single_action5]
	# for modelf in a:
	# 	t = time.perf_counter()
	# 	for i in range(1000):
	# 		modelf(rand_state5())
	# 	print(time.perf_counter() - t)

	# model_fn = get_good_nonlinear_fit(speed_tradeoff=True)
	# model_fn = get_good_nonlinear_fit(speed_tradeoff=False)
	# model_fn = get_nonlinear_fit(2**13, 2**11)
	# save_model_function(model_fn, "nonlin_13_11")
	# model_fn = get_nonlinear_fit(2**15, 2**13)
	# save_model_function(model_fn, "nonlin_15_13")
	# model_fn = get_nonlinear_fit(2**16, 2**12)
	# save_model_function(model_fn, "nonlin_16_12")
	# model_fn = get_nonlinear_fit(2**16, 2**14)
	# save_model_function(model_fn, "nonlin_16_14")
	# model_fn = get_nonlinear_fit(2**18, 2**16)
	# save_model_function(model_fn, "nonlin_18_16")

	# exit()
