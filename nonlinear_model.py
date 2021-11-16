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



# this function is an absolute state and very badly named but I'm in too deep
def optimise_nonlinear_fit(X, Y, test_X, test_Y, scan_vars, scan_ranges, fixed_vars, plot=True, log_scale=True, fixed_colours=True, force_colour=None, show_fixed=True, show_min=False, time_ax=None, override_label=None, show_train=True, noiseless_test=None):

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
		y3 = np.zeros_like(scan_range)
		for i, s in enumerate(scan_range):

			curr_vars[s_idx] = s

			t0 = time.perf_counter()

			fit_fn = nonlinear_fit(X, Y, curr_vars[5], curr_vars[:5], int(curr_vars[6]))

			t1 = time.perf_counter()

			train = fit_errors(X, Y, fit_fn, int(curr_vars[6]))
			test = data_errors(test_X, test_Y, fit_fn)

			if noiseless_test is not None:
				test1 = data_errors(noiseless_test[0], noiseless_test[1], fit_fn)
				y3[i] = test1
		
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
				if noiseless_test is not None:
					plt.plot(scan_range, y3, lw=1, c=colours[s_idx])
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
					p = plt.plot(scan_range, y)#, label=label)
				if show_train:
					plt.plot(scan_range, y1, "--", lw=1, c=p[0].get_color())#, label=f"train, {s_idx}")
				if noiseless_test is not None:
					plt.plot(scan_range, y3, lw=1, c=p[0].get_color())
				if show_fixed: 
					plt.scatter(scan_range[fixed_idx], y[fixed_idx], marker="o", c=p[0].get_color())
				if show_min: 
					plt.scatter(scan_range[min_idx], y[min_idx], marker="x", c=p[0].get_color())
				if time_ax is not None:
					time_ax.plot(scan_range, y2, ":", c=p[0].get_color())

			if log_scale:
				plt.semilogx()
			# plt.legend()

	if noiseless_test is not None:
		return np.array(optima), optimal_score, y, y1, y2, y3
	return np.array(optima), optimal_score, y, y1, y2


def model_fit_score(update_fn, N=2**14):
	X, Y = target_training_data(N, sobol=False)
	return data_errors(X, Y, update_fn)



def evaluate_nonlinear_fit_score(X, Y, test_X, test_Y, params, ntestX=None, ntestY=None):

	t0 = time.perf_counter()
	fit_fn = nonlinear_fit(X, Y, params[5], params[:5], int(params[6]))
	t1 = time.perf_counter()

	train = fit_errors(X, Y, fit_fn, int(params[6]))
	test = data_errors(test_X, test_Y, fit_fn)
	
	if ntestX is not None:
		ntest = data_errors(ntestX, ntestY, fit_fn)
		return test, train, ntest, t1

	else:
		return test, train, t1




def get_train_and_test_data(N_train, N_test, obs_noise=None, dyn_noise=None, ret_noiseless=False):

	X, Y, C = nonlinear_training_data(N_train, sobol=True, incl_f=True, obs_noise=obs_noise, dyn_noise=dyn_noise)
	
	set_dynamic_noise(dyn_noise)
	test_X, test_Y = target_training_data(N_test, sobol=False, incl_f=True, range_factor=1.0)
	set_dynamic_noise(0.)
	test_X, test_Y = corrupt(test_X, test_Y, obs_noise=obs_noise)
	test_Y -= test_X @ C.T

	if ret_noiseless:
		ntest_X, ntest_Y = target_training_data(N_test, sobol=False, incl_f=True, range_factor=1.0)
		ntest_Y -= ntest_X @ C.T
		return X, Y, test_X, test_Y, ntest_X, ntest_Y

	else:
		return X, Y, test_X, test_Y




def scores(M, reg, scales, N_train, N_test, obs_noise=None, dyn_noise=None):

	X, Y, tX, tY, ntX, ntY = get_train_and_test_data(N_train, N_test, obs_noise=None, dyn_noise=None, ret_noiseless=True)
	return evaluate_nonlinear_fit_score(X, Y, tX, tY, list(scales) + [reg, M], ntX, ntY)



from matplotlib import rcParams as mpl_rcp
from matplotlib import cycler as mpl_cycler

def param_graphs(obs_f=0.0, dyn_f=0.0):
	
	obs_noise = obs_f*P_RANGE5
	SCAN_N = 20

	# for the first few:

	X, Y, test_X, test_Y = get_train_and_test_data(2**9, 2**12, obs_noise=obs_noise, dyn_noise=dyn_f)

	nX, nY, ntX, ntY = get_train_and_test_data(2**9, 2**12)
	noiseless_test = [ntX, ntY]
	

	params = list(INITIAL_GUESS) + [1e-3, 256]



	# lambda plots vs noise level for observation random noise
	def lambda_plot(ax=None, N_exp=9):
		print("lambda scans!")
		if ax is None:
			plt.figure()
		else:
			plt.sca(ax)
		axis = np.power(10, np.linspace(-4, 2, 10))

		cmap = cm.get_cmap("hsv", 256)
		cmap = cmap(np.linspace(.3, 1, len(axis)))
		mpl_rcp['axes.prop_cycle'] = mpl_cycler(color=cmap) 
		noise_axis = np.linspace(0, .25, 6)

		for noise_frac in np.linspace(0, .25, 6):
			X, Y, test_X, test_Y = get_train_and_test_data(2**N_exp, 2**12, obs_noise=None, dyn_noise=noise_frac*20)#noise_frac*P_RANGE5
		
			X1, Y1, noiseless_test_X, noiseless_test_Y  = get_train_and_test_data(2**N_exp, 2**12)
			noiseless_test = [noiseless_test_X, noiseless_test_Y]


			ret = optimise_nonlinear_fit(X, Y, test_X, test_Y, [5], [axis], 
											params, False, True, fixed_colours=False, show_min=True, show_fixed=False,
											override_label="$\sigma_{dyn}=" + f"{noise_frac:.2f}"+ "$",
											noiseless_test=noiseless_test)

			opt, opt_s, test, train, t, test_noiseless = ret
			p = plt.plot(axis, test, "--", lw=1)
			plt.plot(axis, test_noiseless, c=p[0].get_color())
			plt.semilogx()


		if ax is None:
			plt.show()


	def more_lambda():

		plt.figure()
		for noise_frac in np.linspace(0, .25, 6):
			plt.plot(np.nan, np.nan, label="$\sigma_{dyn}=" + f"{noise_frac*20:.2f}"+ "$")
		# plt.plot(np.nan, np.nan, "k", label="Noisy test RMSE")
		# plt.plot(np.nan, np.nan, "k--", label="Noisy train RMSE")
		# plt.plot(np.nan, np.nan, "k-:", label="Noiseless test RMSE")

		plt.legend()

		# lambda_plot(N_exp=9)
		# lambda_plot(N_exp=13)
		# lambda_plot(N_exp=17)



		fig, axs = plt.subplots(2, 1, sharex=True)
		lambda_plot(axs[0], 9)
		lambda_plot(axs[1], 14)

		plt.show()
		return

	# more_lambda()


	def param_scans():
		print("parameter scans!")
		params_guess = list(INITIAL_GUESS) + [1e-4, 256]
		# params = list(GOOD_PARAMS) + [GOOD_REG, 512]

		# lambda scan with inital guess
		if False:
			plt.figure()
			X, Y, test_X, test_Y = get_train_and_test_data(2**11, 2**12, obs_noise=obs_noise, dyn_noise=dyn_f)


			ret = optimise_nonlinear_fit(X, Y, test_X, test_Y, [5], [np.power(10, np.linspace(-6, -1, 50))], 
											params_guess, True, True, "$\lambda$")

			# plt.plot(t, )

			# optima = optimise_nonlinear_fit(X, Y, test_X, test_Y, [5], [np.power(10, np.linspace(-6, -1, 50))], 
											# params, True, True, fixed_colours=False, force_colour="tab:pink", override_label="$\lambda$, ")


			plt.show()

		# scans of scale parameters
		if True:
			X, Y, test_X, test_Y = get_train_and_test_data(2**11, 2**12, obs_noise=obs_noise, dyn_noise=dyn_f)
			# X, Y, C = nonlinear_training_data(2**11, sobol=True, incl_f=True, dyn_noise=dyn_noise)#  obs_noise=obs_noise, 

			# set_dynamic_noise(dyn_noise)
			# noiseless_test_X, noiseless_test_Y = target_training_data(2**12, sobol=False, incl_f=True)
			# set_dynamic_noise(0.)
			# test_X, test_Y = corrupt(noiseless_test_X, noiseless_test_Y, obs_noise=obs_noise)
			# test_Y -= test_X @ C.T

			plt.figure()
			scan_range = np.power(10, np.linspace(-2, 3.1, 20))
			scan_ranges = [scan_range for i in range(5)]
			plt.title("$\sigma_{dyn}=" + f"{dyn_f}$")
			optimise_nonlinear_fit(X, Y, test_X, test_Y, range(5), scan_ranges, params_guess, True, True)
		
		# plt.show()


	# param_scans()
	# return

	def N_M_scans():


		# N, M scans, showing convergence and time taken
		# kind of a mess
		params = list(GOOD_PARAMS) + [GOOD_REG, 1024]
		if False:

			exps = np.arange(6, 14)

			cmap = cm.get_cmap("hsv", 256)
			cmap = cmap(np.linspace(.3, 1, len(exps)))
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
				X, Y, C = nonlinear_training_data(int(2**exp), sobol=True, incl_f=True, obs_noise=obs_noise)

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

		if True:

			M_exps = [9, 10, 11]
			N_exps = [11,12,13,14,15,16,17,18]


			cmap = cm.get_cmap("hsv", 256)
			cmap = cmap(np.linspace(.3, 1, len(M_exps)))
			mpl_rcp['axes.prop_cycle'] = mpl_cycler(color=cmap) 

			markers = ["o", "x", "s"]
			def get_m(i):
				return markers[i % len(markers)]

			plt.figure()

			M_lines = [[] for x in M_exps]
			M_lines1 = [[] for x in M_exps]
			for i, exp in enumerate(N_exps): # best number is 10, takes a long time from 11. M should be as high as possible

				X, Y, C = nonlinear_training_data(int(2**exp), sobol=True, incl_f=True, obs_noise=obs_noise, dyn_noise=dyn_f)

				try:
					max_M = M_exps.index(exp)
				except:
					max_M=len(M_exps)

				if max_M == 0:
					continue

				M_axis = np.power(2., M_exps)[:max_M]

				# print(exp, max_M, M_axis)

				ret =  optimise_nonlinear_fit(X, Y, test_X, test_Y, [6], [M_axis], 
											params, False, True, False, show_fixed=False, show_min=False, 
											time_ax=None, override_label=int(2**exp),
											noiseless_test=noiseless_test)
				opt, opt_s, test, train, t, test_noiseless = ret

				for j in range(max_M):
					M_lines[j].append(test_noiseless[j])
					M_lines1[j].append(test[j])
				for j in range(max_M, len(M_exps)):
					M_lines[j].append(np.nan)
					M_lines1[j].append(np.nan)

			for i in range(len(M_lines)):
				print(M_lines[i])
				print(M_lines1[i])

			# best = [0.095]
			best = [0.17, 0.095, 0.075]


			for j, M_exp in enumerate(M_exps):
				try:
					print(j, M_exp, N_exps[j:])
					p = plt.plot(np.power(2., N_exps[j:]), M_lines[j][j:], label=f"M={int(2**M_exp)}")
					plt.plot(np.power(2., N_exps[j:]), M_lines1[j][j:], "--", c=p[0].get_color())
					plt.scatter(np.power(2., N_exps[j:]), M_lines[j][j:], marker="x", c="k", zorder=4)
					plt.plot(np.power(2., N_exps[j:]), [best[j]]*len(M_lines[j][j:]), "k--")
				except Exception as err:
					print(err)
				print(M_lines[j][-1])

			plt.legend()
			plt.semilogx()
			plt.title("Fit quality vs N for various M, $\sigma_{dyn}=" + f"{dyn_f}" + "$")
			plt.ylabel("Test data RMSE")
			plt.xlabel("N")

			# plt.show()

		if False:
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

				X, Y, C = nonlinear_training_data(N, sobol=True, incl_f=True, obs_noise=obs_noise)

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

	N_M_scans()
	return 


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


# INITIAL_GUESS =  [10**3,10**3,1,3,20] 
# default start_params [1e4, 1e4, 1.8, 8.64, 14.4, -4]
def actually_optimise_params_noiseless(M, start_params=INITIAL_GUESS + [-2], to_vary=(2,3,4,5), train_N=2**10, test_N=2**14, obs_f=None, dyn_f=None, print_res=True, return_function=False):

	# params = 0-4: Component of state vector
	# params = 5: regularisation


	obs_noise = obs_f*P_RANGE5

	X, Y, test_X, test_Y = get_train_and_test_data(train_N, test_N, obs_noise=obs_f*P_RANGE5, dyn_noise=dyn_f)


	np.set_printoptions(formatter={'float': lambda x: "{0:0.4f}".format(x)})

	params0 = np.array([start_params[i] for i in to_vary])
	print(params0)


	def param_lookup(varying_params):
		return [varying_params[to_vary.index(i)] if i in to_vary else start_params[i] for i in range(6)]


	def to_optimise(params):
		params = param_lookup(params) 
		return data_errors(test_X, test_Y, nonlinear_fit(X, Y, np.power(10., params[5]), params[:5], M))
	

	def cb(x):
		print("\r" + str(x), end="")


	res = scipy.optimize.minimize(to_optimise, params0, method="Nelder-Mead", callback=cb,
								options={"maxiter" : 3000, "fatol" : 1e-4})

	if print_res:
		print(param_lookup(res.x), res.fun)

	params1 = param_lookup(res.x)
	fn = nonlinear_fit(X, Y, np.power(10., params1[5]), params1[:5], M)
	nX, nY, ntX, ntY = get_train_and_test_data(train_N, test_N, obs_noise=None)
	noiseless_fit = data_errors(ntX, ntY, fn)

	if return_function:
		return fn, params1, res.fun, noiseless_fit, res.message, res.nfev
	else:
		return params1, res.fun, noiseless_fit, res.message, res.nfev


def create_optimal_fit(M, train_N=2**10, obs_f=None):
	ret = actually_optimise_params_noiseless(M, train_N=train_N, obs_f=obs_f, return_function=True)
	return ret[0], ret[1], ret[2]


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



def create_optimal_noisy_fits():

	M_exp = 10 #11

	sigma_obs_ax = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.25, .5]
	sigma_dyn_ax = [0.02, 0.05, 0.1, 0.2, 0.5, 1., 2., 5., 10.]




	params = [1e5, 1e5, .3, 4.0, 20., -3]
	results = []

	if False:
		for i, noise in enumerate(sigma_obs_ax):
		# for i, noise in enumerate(sigma_dyn_ax):
			M, N = 2**M_exp, 2**(M_exp+6)


			# params1, res.fun, noiseless_fit, res.message, res.nfev
			params, fit1, fit2, msg, evals = actually_optimise_params_noiseless(M, params, train_N=N, 
													print_res=False, return_function=False, 
													obs_f=noise, dyn_f=0.0)
													# obs_f=0.0, dyn_f=noise)

			results.append([params, fit1, fit2, msg, evals])

			print("results so far: ###########################")
			for row in results:
				print(*row)
			print("getting fit.")

			noise_str = str(sigma_obs_ax[i]).replace(".", "_")
			# noise_str = str(sigma_dyn_ax[i]).replace(".", "_")
			print(noise_str)
			# save_model_function(get_nonlinear_fit(N, M, params=params, obs_noise=noise*P_RANGE5), f"nonlin_noisy_{M_exp}_{M_exp+6}_obs_{noise_str}")
			save_model_function(get_nonlinear_fit(N, M, params=params, dyn_noise=noise), f"nonlin_noisy_{M_exp}_{M_exp+6}_dyn_{noise_str}")



	sigma_obs_ps = [[100000.0, 100000.0, 0.20746952547879569, 3.2255525261913265, 16.401264590483326, -8.223714740430797],
					[100000.0, 100000.0, 0.20685391185848268, 3.4583040948588812, 15.489906229927175, -8.54559518759132],
					[100000.0, 100000.0, 0.21582199663233603, 3.3384907914411492, 15.936256035266862, -8.47508444907799],
					[100000.0, 100000.0, 0.21532468122914722, 3.3508991449190475, 15.27284273364794, -8.803506958996023],
					[100000.0, 100000.0, 0.21285916402516986, 3.529043179474634, 15.46340659202623, -8.978634859523698],
					[100000.0, 100000.0, 0.29156506735439697, 4.183415391114394, 10.20485977879181, -9.511142500205942],
					[100000.0, 100000.0, 0.25842825532785985, 5.734622514373541, 7.427117550967562, -9.086263301420786],
					[100000.0, 100000.0, 1.0015617783412678, 11.08033575029842, 28.866734093942163, 0.31679770509720817],
					[100000.0, 100000.0, 2.8124106999568834, 33.8133552445302, 44.46308125717164, -0.065959322556886]]

	sigma_obs_fits = [[0.07318235864465265, 0.07150576129966375],
						[0.07998015509134292, 0.07022527828523058],
						[0.11882333621550238, 0.07082361050273389],
						[0.20486119325478477, 0.07459120478414348],
						[0.3893607006107528, 0.08650403375292093],
						[0.9473045677832639, 0.1929175188176782],
						[1.8412787443006478, 0.4334781600969593],
						[4.034310917560231, 1.4325881438012071],
						[6.930579469992963, 2.9424068376384036]]

	sigma_dyn_ps = [[100000.0, 100000.0, 0.2042242218559962, 3.349669637243748, 15.884470236191232, -7.021705173068747],
					[100000.0, 100000.0, 0.2048954541399077, 3.445830003179048, 15.728892768505286, -7.027785757770307],
					[100000.0, 100000.0, 0.19736902431809022, 3.452992588900665, 16.20334891586571, -7.054462954962145],
					[100000.0, 100000.0, 0.20139939205666557, 3.523906721092351, 15.586296986927099, -7.06021451883896],
					[100000.0, 100000.0, 0.20944345774225165, 3.2431776433357915, 16.41344636048393, -6.9404370757248115],
					[100000.0, 100000.0, 0.2677234399328853, 3.8347752126643866, 14.733759309513358, -6.289800515604641],
					[100000.0, 100000.0, 0.2539708372201905, 3.963872806845731, 16.245235686813377, -5.658471932698257],
					[100000.0, 100000.0, 0.38749677802214355, 9.369896177682845, 14.308866078627492, 0.12481871879545212],
					[100000.0, 100000.0, 0.28051539942251996, 11.096214694298727, 17.58139945546847, 1.3472651711245796]]

	sigma_dyn_fits = [[0.07205894063041607, 0.07065924731072026],
						[0.07708657101948731, 0.07047070899440674],
						[0.09669914872855934, 0.07171334553534904],
						[0.14887480647381582, 0.0730995211646937],
						[0.3376126521622893, 0.081033468551896],
						[0.6581310968946876, 0.11143499651474775],
						[1.326066827222177, 0.20290882483087425],
						[3.2589424588435447, 0.4591737352279506],
						[6.392704063601857, 0.8998807363353406]]



	plt.plot(np.arange(9), [x[1] for x in sigma_dyn_fits])
	plt.show()

	for i, ps in enumerate(sigma_dyn_ps):
		fit_fn = get_nonlinear_fit(2**16, 2**10, params=ps, dyn_noise=sigma_dyn_ax[i])

		noise_str = str(sigma_dyn_ax[i]).replace(".", "_")
		fname = f"nonlin_noisy_10_16_dyn_{noise_str}"
		save_model_function(fit_fn, fname)
	# for row in ...
	# save ACTUAL noisy fit.



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


def manual_noisy_convergence_plots():

	sigma_obs_ax = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.25, .5]
	sigma_dyn_ax = [0.02, 0.05, 0.1, 0.2, 0.5, 1., 2., 5., 10.]

	M_9_obs  = np.array([3, 3, 3, 5, 6, 6, 6, 6, 6])
	M_10_obs = np.array([3, 3, 4, 6, 6, 6, 6, 5, 6])
	M_11_obs = np.array([3, 4, 4, 6, 6, 6, 5, 5, 5])

	M_9_dyn  = np.array([3, 3, 3, 3, 4, 6, 6, 7, 7])
	M_10_dyn = np.array([3, 3, 3, 4, 5, 6, 6, 6, 6])
	M_11_dyn = np.array([3, 3, 4, 5, 5, 5, 6, 6, 6])

	cmap = cm.get_cmap("hsv", 256)
	cmap = cmap(np.linspace(.3, 1, 3))
	mpl_rcp['axes.prop_cycle'] = mpl_cycler(color=cmap) 


	fig, axs = plt.subplots(1,2)

	plt.suptitle("N value required for convergence vs. noise", y=.95)

	plt.sca(axs[0])
	plt.semilogx()
	plt.xticks(sigma_obs_ax)

	plt.ylabel("$\log_2{(N/M)}$")
	plt.xlabel("$\sigma_{obs}$")

	plt.plot(sigma_obs_ax, M_9_obs, ls="", marker="o", label="M=512")
	plt.plot(sigma_obs_ax, M_10_obs, ls="", marker="^", label="M=1024")
	plt.plot(sigma_obs_ax, M_11_obs, ls="", marker="x", label="M=2048")
	plt.plot(sigma_obs_ax, np.mean([M_9_obs, M_10_obs, M_11_obs], axis=0), "k--")
	plt.legend()

	plt.sca(axs[1])
	plt.semilogx()
	plt.xticks(sigma_dyn_ax)

	plt.ylabel("$\log_2{(N/M)}$")
	plt.xlabel("$\sigma_{dyn}$")

	plt.plot(sigma_dyn_ax, M_9_dyn, ls="", marker="o", label="M=512")
	plt.plot(sigma_dyn_ax, M_10_dyn, ls="", marker="^", label="M=1024")
	plt.plot(sigma_dyn_ax, M_11_dyn, ls="", marker="x", label="M=2048")
	plt.plot(sigma_dyn_ax, np.mean([M_9_dyn, M_10_dyn, M_11_dyn], axis=0), "k--")
	plt.legend()

	plt.show()



if __name__ == "__main__":

	# create_optimal_noisy_fits()

	while True:
		# plot_scan_matrix(get_good_linear_fit(enforce_constraints=False), axs_in=ax)

		start_state = rand_state5()
		ax = plot_scan_matrix(target, start_state=start_state)
		model_fn = get_optimal_nonlin_fit(12)
		plot_scan_matrix(model_fn, axs_in=ax, start_state=start_state)
		plt.suptitle("Nonlinear fit, single-variable scans, M=12")
		plt.show()

	exit()
	# manual_noisy_convergence_plots()

	# create_optimal_nonlin_fits()
	# exit()

	# nonlinear_model_slice()


	# IN = [1e5, 1e5, 2.05206813e-01,  3.52071053e+00, 1.58185446e+01, -4.04811817e+00]
	# M = 1024
	# GOOD_PARAMS + [GOOD_REG]
	# actually_optimise_params_noiseless(M, IN, (2,3,4,5), train_N=16*M)


	# axis = np.linspace(0, 0.01, 10)
	# errors = []
	# for dyn_f in axis:
	# 	N_test = int((2**14)*(1 + 100*dyn_f*dyn_f))
	# 	print(N_test)
	# 	test, train, ntest, t = scores(2**6, GOOD_REG, GOOD_PARAMS, int(N_test), 2**12, obs_noise=None, dyn_noise=dyn_f)
	# 	errors.append(ntest)
	# plt.plot(axis, errors)
	# plt.show()
	# exit()

	if False:
		# param_graphs(0.0, 0.0)#0.05*20)
		param_graphs(0.0, .005*20)#0.05*20)
		param_graphs(0.0, .01*20)#0.05*20)
		param_graphs(0.0, .025*20)#0.05*20)
		param_graphs(0.0, .05*20)#0.05*20)
		param_graphs(0.0, .1*20)#0.05*20)
		param_graphs(0.0, .25*20)#0.05*20)
		param_graphs(0.0, .5*20)#0.05*20)
		# param_graphs(0.0, 0.05*20)#0.05*20)
		# param_graphs(0.0, 0.1*20)#0.05*20)
		plt.show()

		exit()



	# param_graphs(0.000)
	# plt.show()
	param_graphs(0.1)
	# plt.show()

	param_graphs(0.25)
	# plt.show()

	param_graphs(0.5)
	# plt.show()

	# param_graphs(0.05)
	plt.show()

	exit()



	M=int(2**8)
	nfs = np.linspace(0, 0.2, 10)
	data = []
	params = INITIAL_GUESS + [-2]
	for noise_frac in nfs:
		ret = actually_optimise_params_noiseless(M, start_params=params, train_N=2**10, obs_f=noise_frac, return_function=True)

		fn, ps, fit, noiseless_fit = ret[:4]
		params = ps
		print(fit, noiseless_fit)

		data.append({"obs_f" : noise_frac, "params" : ps, "fit_fn" : fn, "noisy_fit": fit, "fit" : noiseless_fit})


	save_data(data, "M=2^8_noisy_nonlin_fits")# and N=2^10
	load_data(data, "M=2^8_noisy_nonlin_fits")

	plt.plot(nfs, [x["noisy_fit"] for x in data])
	plt.plot(nfs, [x["fit"] for x in data])
	plt.show()




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
