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

fast_kfn4 = lambda x, y, s: np.exp(-(
	# np.square((x[0]-y[:,0])*INV_SQRT2/s[0]) +
	# np.square((x[1]-y[:,1])*INV_SQRT2/s[1]) +
	np.square(np.sin(.5*(x[2]-y[:,2]))*INV_SQRT2/s[2]) +
	np.square((x[3]-y[:,3])*INV_SQRT2/s[3])
))

fast_kfn5 = lambda x, y, s: np.exp(-(
	# np.square((x[0]-y[:,0])*INV_SQRT2/s[0]) +
	# np.square((x[1]-y[:,1])*INV_SQRT2/s[1]) +
	np.square(np.sin(.5*(x[2]-y[:,2]))*INV_SQRT2/s[2]) +
	np.square((x[3]-y[:,3])*INV_SQRT2/s[3]) +
	np.square((x[4]-y[:,4])*INV_SQRT2/s[4])
))



def nonlinear_training_data(N=512, t_step=0.2, sobol=True, incl_f=True, **kwargs):

	data, C = linear_fit(N, t_step, sobol, return_data=True, incl_f=incl_f, enforce_constraints=True, **kwargs)

	# err = (Y.T - C @ X.T).T
	err = (data[3] - data[2] @ C.T)

	# print(Y)
	# print(err)
	# print(np.linalg.norm(Y, "fro"))
	# print(np.linalg.norm(err, "fro"))

	return data[2], err, C

warned_fast_version=False
def warn_fast_version():
	global warned_fast_version
	if not warned_fast_version:
		warned_fast_version=True
		print("note: Using accelerated kfn without first 2 elements due to large scales.")


def nonlinear_fit(X, Y, reg, scales, M=256, t_step=0.2):

	N = len(X)
	if N < M:
		raise Exception(f"N={N} but M={M}!")

	λ = reg
	σ = scales

	K_MM = np.zeros((M, M))
	K_MN = np.zeros((M, N))

	if X.shape[1] == 4:
		if scales[0] >= 1e3 and scales[1] >= 1e3:
			kfn = fast_kfn4
			warn_fast_version()
		else:
			kfn = kfn4
	else:
		if scales[0] >= 1e3 and scales[1] >= 1e3:
			kfn = fast_kfn5
			warn_fast_version()
		else:
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


# def fit_errors(X, Y, fit_fn, M):
# 	Y_est = [fit_fn(x) for x in X[:M]]
# 	errors = np.sum(np.abs(Y[:M] - Y_est), axis=1)
# 	return np.std(errors)


# def data_errors(X, Y, fit_fn):
# 	return np.std(np.sum(np.abs(Y - [fit_fn(x) for x in X]), axis=1))

def fit_errors(X, Y, fit_fn, M):
	return np.sqrt(np.sum(np.square(Y[:M] - [fit_fn(x) for x in X[:M]]))/(M*Y.shape[1]))

def data_errors(X, Y, fit_fn):
	return np.sqrt(np.sum(np.square(Y - [fit_fn(x) for x in X]))/(Y.shape[0]*Y.shape[1]))



INITIAL_GUESS = [10**3,10**3,1,3,20] 
# GOOD_PARAMS = [1000, 1000, 0.26605725, 4.37394821, 16.17130779]
# [1e4, 1e4, 0.21105305,  3.83526513, 19.50607067] better for M=512
# [1e4, 1e4, 0.50482846,  7.68111386, 22.60525297, 1e-4.52513962]
# [ 0.26605725,  4.37394821, 16.17130779, -2.9287668 ]

# For 512, 2048
# GOOD_PARAMS = [ 4.08837602e+02,  5.41441326e+02,  2.67396016e-01,  3.99692764e+00, 1.97575870e+01]
# GOOD_REG = np.power(10, -2.87837590)

# For 1024, 16000ish
GOOD_PARAMS = [1e5, 1e5, 0.203, 3.433, 15.848]
GOOD_REG = np.power(10, -7.358)



def get_nonlinear_fit(N, M, ret_full_fit=True, return_all=False, incl_f=True, params=None, **kwargs):#, obs_noise=None, obs_bias=None, dyn_noise=None, dyn_bias=None):
	print("fitting...")
	
	X, Y, C = nonlinear_training_data(N, sobol=True, incl_f=incl_f, **kwargs)# this is where noise is inserted

	perm = np.random.permutation(np.arange(N))
	centre_idxs = perm[:M]
	other_idxs = perm[M:]
	X = X[perm, :]
	Y = Y[perm, :]

	if params is None:
		REG = GOOD_REG
		good_params = GOOD_PARAMS
	else:
		good_params = params[:5]
		REG = params[5]
		if REG < 0:
			REG = np.power(10., REG)

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


def get_good_nonlinear_fit(M=None, speed_tradeoff=True, ret_full_fit=True, return_all=False, incl_f=True, **kwargs):
	
	if M is not None:
		return get_nonlinear_fit(16*M, M, ret_full_fit, return_all, incl_f, **kwargs)

	else:
		if speed_tradeoff:
			N = 2**11
			M = 2**9
		else:
			N = 2**13
			M = 2**10

		return get_nonlinear_fit(N, M, ret_full_fit, return_all, incl_f, **kwargs)



def get_good_noisy_nonlinear_fit(noise_fraction, N=2**12, M=2**9, incl_f=True):

	return get_nonlinear_fit(N, M, incl_f=incl_f, obs_noise=noise_fraction*P_RANGE5, dyn_noise=None)



def nonlinear_model_slice():

	# X, Y = target_training_data(256, sobol=True)
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




def optimise_nonlinear_fit(X, Y, test_X, test_Y, scan_vars, scan_ranges, fixed_vars, plot=True, log_scale=True, fixed_colours=True, force_colour=None, show_fixed=True, show_min=False, time_ax=None, override_label=None, show_train=True):

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
				if show_train:
					plt.plot(scan_range, y1, "--", lw=1, c=colours[s_idx])#, label=f"train, {s_idx}")
				if show_fixed: 
					plt.scatter(scan_range[fixed_idx], y[fixed_idx], marker="o", c=colours[s_idx])
				if show_min: 
					plt.scatter(scan_range[min_idx], y[min_idx], marker="x", c=colours[s_idx])
				if time_ax is not None:
					time_ax.plot(scan_range, y2, ":", c=colours[s_idx])
			else:
				if force_colour is not None:
					p = plt.plot(scan_range, y, c=force_colour, label=label)
				else:
					p = plt.plot(scan_range, y, c=colours[s_idx], label=label)
				if show_train:
					plt.plot(scan_range, y1, "--", lw=1, c=p[0].get_color())#, label=f"train, {s_idx}")
				if show_fixed: 
					plt.scatter(scan_range[fixed_idx], y[fixed_idx], marker="o", c=p[0].get_color())
				if show_min: 
					plt.scatter(scan_range[min_idx], y[min_idx], marker="x", c=p[0].get_color())
				if time_ax is not None:
					time_ax.plot(scan_range, y2, ":", c=p[0].get_color())

			if log_scale:
				plt.semilogx()
			plt.legend()

	return np.array(optima), optimal_score, y, y1, y2


def model_fit_score(update_fn, N=2**14):
	X, Y = target_training_data(N, sobol=False)
	return data_errors(X, Y, update_fn)



def evaluate_nonlinear_fit_score(X, Y, test_X, test_Y, params):

	t0 = time.perf_counter()
	fit_fn = nonlinear_fit(X, Y, params[5], params[:5], int(params[6]))
	t1 = time.perf_counter()

	train = fit_errors(X, Y, fit_fn, int(params[6]))
	test = data_errors(test_X, test_Y, fit_fn)
	
	return test, train, t1



def get_train_and_test_data(N_train, N_test, obs_noise=None):

	X, Y, C = nonlinear_training_data(N_train, sobol=True, incl_f=True, obs_noise=None)
	
	test_X, test_Y = target_training_data(N_test, sobol=False, incl_f=True)
	test_X, test_Y = corrupt(test_X, test_Y, obs_noise=None)
	test_Y -= test_X @ C.T

	return X, Y, test_X, test_Y



def param_graphs(noise_fraction=0.0):
	
	obs_noise = noise_fraction*P_RANGE5
	SCAN_N = 20

	# for the first few:
	X, Y, test_X, test_Y = get_train_and_test_data(2**9, 2**12, obs_noise=obs_noise)
	params = list(GOOD_PARAMS) + [1e-3, 256]


	# lambda plots vs noise level for observation random noise
	if False:
		plt.figure()
		for noise_frac in np.linspace(0, .1, 7):
			X, Y, test_X, test_Y = get_train_and_test_data(2**11, 2**12, obs_noise=noise_fraction*P_RANGE5)

			optima = optimise_nonlinear_fit(X, Y, test_X, test_Y, [5], [np.power(10, np.linspace(-4, 0, 50))], 
											params, True, True, fixed_colours=False)
		plt.show()


	def param_scans():
		params_guess = list(INITIAL_GUESS) + [1e-4, 512]
		params = list(GOOD_PARAMS) + [GOOD_REG, 512]

		# lambda scan with inital guess
		if True:
			plt.figure()
			X, Y, test_X, test_Y = get_train_and_test_data(2**11, 2**12)


			optima = optimise_nonlinear_fit(X, Y, test_X, test_Y, [5], [np.power(10, np.linspace(-6, -1, 50))], 
											params_guess, True, True, "$\lambda$")

			# optima = optimise_nonlinear_fit(X, Y, test_X, test_Y, [5], [np.power(10, np.linspace(-6, -1, 50))], 
											# params, True, True, fixed_colours=False, force_colour="tab:pink", override_label="$\lambda$, ")


			plt.show()

		# scans of scale parameters
		if True:
			X, Y, C = nonlinear_training_data(2**11, sobol=True, incl_f=True, obs_noise=obs_noise)

			test_X, test_Y = target_training_data(2**12, sobol=False, incl_f=True)
			test_X, test_Y = corrupt(test_X, test_Y, obs_noise=obs_noise)
			test_Y -= test_X @ C.T

			plt.figure()
			scan_range = np.power(10, np.linspace(-2, 3.1, 50))
			scan_ranges = [scan_range for i in range(5)]

			optimise_nonlinear_fit(X, Y, test_X, test_Y, range(5), scan_ranges, params_guess, True, True)
		
		plt.show()


	param_scans()

	def N_M_scans():


		# N, M scans, showing convergence and time taken
		# kind of a mess
		params = list(GOOD_PARAMS) + [GOOD_REG, 1024]
		if False:

			exps = np.arange(6, 14)

			cmap = cm.get_cmap("hsv", 256)
			cmap = cmap(np.linspace(.3, 1, len(exps)))
			from matplotlib import rcParams as mpl_rcp
			from matplotlib import cycler as mpl_cycler
			mpl_rcp['axes.prop_cycle'] = mpl_cycler(color=cmap) 

			markers = ["o", "x", "s"]
			def get_m(i):
				return markers[i % len(markers)]

			fig1 = plt.figure()
			plt.title("Fit quality vs M for various N values")
			ax1 = plt.gca()
			fig2 = plt.figure()
			ax2 = plt.gca()
			vals = [6,7,8,9,10,11,12,13]

			M_lines = {i:[[], []] for i in [4,5,6,7,8,9,10,11]}

			for i, exp in enumerate(vals): # best number is 10, takes a long time from 11. M should be as high as possible
				X, Y, C = nonlinear_training_data(int(2**exp), sobol=True, incl_f=True)

				M_exps = np.linspace(4, exp, 10)
				M_axis = np.power(2, M_exps)

				plt.sca(ax1)
				opt, opt_s, test, train, t =  optimise_nonlinear_fit(X, Y, test_X, test_Y, [6], [M_axis], 
										params, True, True, False, show_fixed=False, show_min=False, time_ax=None, override_label=int(2**exp))

				print(t, test)

				plt.sca(ax2)
				plt.scatter(t, test, marker=get_m(i), label=f"N={int(2**exp)}")


			plt.legend()
			plt.semilogx()
			plt.title("Fit quality vs computation time for various N, M values")
			plt.ylabel("Test data RMSE")
			plt.xlabel("Computation time (s)")
			plt.show()

		if False:

			M_exps = [7,8,9,10]
			N_exps = [7, 8, 9, 10, 11, 12, 13, 14, 15]


			cmap = cm.get_cmap("hsv", 256)
			cmap = cmap(np.linspace(.3, 1, len(M_exps)))
			from matplotlib import rcParams as mpl_rcp
			from matplotlib import cycler as mpl_cycler
			mpl_rcp['axes.prop_cycle'] = mpl_cycler(color=cmap) 

			markers = ["o", "x", "s"]
			def get_m(i):
				return markers[i % len(markers)]

			plt.figure()

			M_lines = [[] for x in M_exps]
			for i, exp in enumerate(N_exps): # best number is 10, takes a long time from 11. M should be as high as possible
				X, Y, C = nonlinear_training_data(int(2**exp), sobol=True, incl_f=True)

				max_M = min(i+1, len(M_exps))
				M_axis = np.power(2., M_exps)[:max_M]

				opt, opt_s, test, train, t =  optimise_nonlinear_fit(X, Y, test_X, test_Y, [6], [M_axis], 
										params, False, True, False, show_fixed=False, show_min=False, time_ax=None, override_label=int(2**exp))

				for j in range(max_M):
					M_lines[j].append(test[j])
				for j in range(max_M, len(M_exps)):
					M_lines[j].append(np.nan)

			for j, M_exp in enumerate(M_exps):
				plt.plot(np.power(2., N_exps[j:]), M_lines[j][j:], label=f"M={int(2**M_exp)}")
				plt.scatter(np.power(2., N_exps[j:]), M_lines[j][j:], marker="x", c="k", zorder=4)
				plt.plot(np.power(2., N_exps[j:]), [M_lines[j][-1]]*len(M_lines[j][j:]), "k--")

			plt.legend()
			plt.semilogx()
			plt.title("Fit quality vs N for various M")
			plt.ylabel("Test data RMSE")
			plt.xlabel("N")
			plt.show()



		if True:
			plt.figure()


			M_exps1 = [7,8,9,10,11,12]
			opt_ts = []
			opt_s = []
			for M_exp in M_exps1:
				opt_fit, props = get_optimal_nonlin_fit(M_exp, return_properties=True)

				t = time.perf_counter()
				for i in range(1000):
					opt_fit([1.,1.,1.,1.,1.])
				t_eval = (time.perf_counter() - t)/1000
				
				opt_ts.append(t_eval)
				opt_s.append(props["fit"])



			M_exps = np.linspace(7, 11, 40)

			t1s = []
			t2s = []
			s = []
			fit_fns = []

			for i, M_exp in enumerate(M_exps):
				N = int(2.**(M_exp+4))

				X, Y, C = nonlinear_training_data(N, sobol=True, incl_f=True)

				t = time.perf_counter()
				fit_fn = nonlinear_fit(X, Y, GOOD_REG, GOOD_PARAMS, int(2**M_exp))
				t_fit = time.perf_counter() - t

				fit_fns.append(fit_fn)

				test = data_errors(test_X, test_Y, fit_fn)
			
				s.append(test)
				t1s.append(t_fit)

			for i, fit_fn in enumerate(fit_fns):
				t = time.perf_counter()
				for i in range(1000):
					fit_fn([1.,1.,1.,1.,1.])
				t_eval = (time.perf_counter() - t)/1000
				t2s.append(t_eval)
			

			plt.loglog()

			plt.xlabel("Test data RMSE")
			plt.ylabel("Fitting time (s)")
			plt.scatter(s,t1s, c="r", marker="x")


			plt.semilogy()
			plt.gca().twinx()
			plt.loglog()


			plt.scatter(opt_s, opt_ts, c="c", marker="*", s=80, label="Evaluation time\n(optimised)", zorder=10)

			plt.ylabel("Evaluation time (s)")
			plt.scatter(s,t2s, c="k", marker="x", label="Evaluation time")
			plt.scatter([np.nan], [np.nan], c="r", marker="x", label="Fitting time")
			plt.semilogy()




			plt.legend(loc=3)
			plt.title("Fit quality vs computation time (N=16M)")
			plt.show()

	# N_M_scans()


	if False:

		plt.figure()
		N_values = np.arange(10.1, 16)
		y_values = np.zeros((len(N_values), 3))
		for i, N in enumerate(N_values):
			print("N =", N)
			X, Y, C = nonlinear_training_data(int(2**N), sobol=True, incl_f=True)
			y_values[i,:] = evaluate_nonlinear_fit_score(X, Y, test_X, test_Y, params)
			print(y_values[i,:])
		plt.plot(N_values, y_values[:,:2])
		ax2=plt.gca().twinx()
		ax2.plot(N_values, y_values[:,2])
		plt.show()






	print(optima)
	params = optima




def actually_optimise_params_noiseless(M, start_params=[1e4, 1e4, 1.8, 8.64, 14.4, -4], to_vary=(2,3,4,5), train_N=2**10, test_N=2**14, print_res=True):

	# params = 0-4: Component of state vector
	# params = 5: regularisation

	X, Y, C = nonlinear_training_data(train_N, sobol=True, incl_f=True)
	test_X, test_Y = target_training_data(test_N, sobol=False, incl_f=True)
	test_Y -= test_X @ C.T


	np.set_printoptions(formatter={'float': lambda x: "{0:0.4f}".format(x)})

	params0 = np.array([start_params[i] for i in to_vary])
	print(params0)


	def param_lookup(varying_params):
		return [varying_params[to_vary.index(i)] if i in to_vary else start_params[i] for i in range(6)]


	def to_optimise(params):
		params = param_lookup(params) 
		return data_errors(test_X, test_Y, nonlinear_fit(X, Y, np.power(10, params[5]), params[:5], M))
	

	def cb(x):
		print("\r" + str(x), end="")


	res = scipy.optimize.minimize(to_optimise, params0, method="Nelder-Mead", callback=cb,
								options={"disp" : True, "maxiter" : 3000, "fatol" : 1e-4})

	if print_res:
		print(res)

	return param_lookup(res.x), res.fun, res.message, res.nfev


def create_optimal_nonlin_fits():

	M_exps = [13, 14]#[5, 6, 7, 8, 9, 10, 11]


	params = opt_results[-1] # [1e5, 1e5, .3, 4.0, 20., -1]
	results = []

	for i, M_exp in enumerate(M_exps):
		M, N = 2**M_exp, 2**(M_exp+4)

		if M_exp <= 12:
			params, f, msg, evals = actually_optimise_params_noiseless(M, params, train_N=N, print_res=False)
			results.append([params, f, msg, evals])

			print("results so far: ###########################")
			for row in results:
				print(*row)
			print("getting fit.")

		save_model_function(get_nonlinear_fit(N, M, params=params), f"nonlin_noiseless_{M_exp}_{M_exp+4}")
		# save_model_function(get_nonlinear_fit(N, M, params=opt_results[i]), f"nonlin_noiseless_{M_exp}_{M_exp+4}")



	# results so far: (5-12) ###########################
	# 100000.0, 100000.0, 2.5636051761293346, 10.194329869708739, 38.60627651216437, -1.4615412974557698] 0.8935181844105925 Optimization terminated successfully. 482
	# [100000.0, 100000.0, 0.5682138518280841, 8.392943330777053, 28.616384098618443, -2.600039477420786] 0.5747402428175139 Optimization terminated successfully. 197
	# [100000.0, 100000.0, 0.38425180501864353, 6.886683497923468, 28.14657106948843, -3.025675652267809] 0.3600858933452549 Optimization terminated successfully. 238
	# [100000.0, 100000.0, 0.30927512963021797, 4.612733144846214, 21.55584277251825, -6.202338442607367] 0.24086718389699033 Optimization terminated successfully. 186
	# [100000.0, 100000.0, 0.2361459904426209, 3.680700742407869, 19.179522533757925, -7.509259600366734] 0.15212843103048765 Optimization terminated successfully. 162
	# [100000.0, 100000.0, 0.20482006775634146, 3.4539124242260906, 15.544205881687411, -9.062133000902449] 0.07123510373679331 Optimization terminated successfully. 147
	# [100000.0, 100000.0, 0.1777849629314689, 2.8184787674962353, 12.736700693041925, -12.222667617861262] 0.03202789664112534 Optimization terminated successfully. 198
	# [100000.0, 100000.0, 0.1351635294710545, 2.2804776402899023, 9.935672409194662, -17.149617495473493] 0.014008829018339204 Optimization terminated successfully. 253



warned_not_optimised = False
def warn_not_optimised(M_exp):
	global warned_not_optimised
	if not warned_not_optimised:
		print(f"M=2**{M_exp}: not an optimised fit")
		warned_not_optimised = True

def get_optimal_nonlin_fit(M_exp, log=False, return_properties=False):

	M_exp = int(M_exp)
	M_exps = [5, 6, 7, 8, 9, 10, 11, 12]


	opt_params = [
	[100000.0, 100000.0, 2.5636051761293346, 10.194329869708739, 38.60627651216437, -1.4615412974557698],
	[100000.0, 100000.0, 0.5682138518280841, 8.392943330777053, 28.616384098618443, -2.600039477420786],
	[100000.0, 100000.0, 0.38425180501864353, 6.886683497923468, 28.14657106948843, -3.025675652267809],
	[100000.0, 100000.0, 0.30927512963021797, 4.612733144846214, 21.55584277251825, -6.202338442607367],
	[100000.0, 100000.0, 0.2361459904426209, 3.680700742407869, 19.179522533757925, -7.509259600366734],
	[100000.0, 100000.0, 0.20482006775634146, 3.4539124242260906, 15.544205881687411, -9.062133000902449],
	[100000.0, 100000.0, 0.1777849629314689, 2.8184787674962353, 12.736700693041925, -12.222667617861262],
	[100000.0, 100000.0, 0.1351635294710545, 2.2804776402899023, 9.935672409194662, -17.149617495473493]]

	fits = [0.8935181844105925, 0.5747402428175139, 0.3600858933452549, 0.24086718389699033, 
			0.15212843103048765, 0.07123510373679331, 0.03202789664112534, 0.014008829018339204,
			0.010451618798334] # this final row for 13, fit to parameters of 12
	
	name = f"nonlin_noiseless_{M_exp}_{M_exp+4}"
	try:
		idx = M_exps.index(M_exp)

		if return_properties:
			return load_model_function(name, log=log), {"fit" : fits[idx], "params" : opt_params[idx]}
		else:
			return load_model_function(name, log=log)
	except:
		warn_not_optimised(M_exp)
		if return_properties:
			return fn, {"fit" : model_fit_score(fn), "params" : opt_params[-1]}
		else:
			return load_model_function(name, log=log)



if __name__ == "__main__":

	# create_optimal_nonlin_fits()
	# exit()

	# nonlinear_model_slice()


	# IN = [1e5, 1e5, 2.05206813e-01,  3.52071053e+00, 1.58185446e+01, -4.04811817e+00]
	# M = 1024
	# GOOD_PARAMS + [GOOD_REG]
	# actually_optimise_params_noiseless(M, IN, (2,3,4,5), train_N=16*M)

	param_graphs(0.0)



	nonlin_fn = get_good_nonlinear_fit(ret_full_fit=True)
	actual_fn = target5


	exit()

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

	model_fn = get_nonlinear_fit(2**13, 2**9)
	save_model_function(model_fn, "nonlin_13_9")
	model_fn = get_nonlinear_fit(2**16, 2**12)
	save_model_function(model_fn, "nonlin_16_12")

	# exit()
