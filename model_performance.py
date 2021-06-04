from globals import *
from utils import *
from simulation_utils import *
from cartpole import *
from linear_model import *
from nonlinear_model import *
from rollouts import *



def model_comparison_scatter(function, NK=100, incl_f=True):

	colours = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

	SCAN_N = 10

	try:
		NK[0]
	except:
		NK = [NK]*5

	print(NK)

	if incl_f:
		bounds = P_RANGE5
	else:
		bounds = P_RANGE4

	# fig, axs = plt.subplots(5, 1)
	# axs = axs.flatten()
	for i in range(len(bounds)):
		# plt.sca(axs[i])
		plt.figure(figsize=(3.4, 2.9))
		a = np.linspace(-10, 10, 2)
		plt.plot(a,a, "k--", alpha=0.5, lw=1)

		for k in range(NK[i]):

			x = rand_state(bounds)
			x1 = x.copy()

			scan = bounds[i] * np.linspace(-1, 1, SCAN_N)
			vals1 = np.zeros((SCAN_N, 4))
			vals2 = np.zeros((SCAN_N, 4))

			for l, s in enumerate(scan):

				x1[i] = s
				vals1[l, :] += target(x1)
				vals2[l, :] += function(x1)

			for j in range(0,4):

				zorder = [7, 5, 6, 4]
				lw = 2
				alpha = 1
				if k == 0:
					p = plt.plot(vals1[:,j], vals2[:,j], c=colours[j], lw=lw, label=VAR_STR[j], alpha=alpha, zorder=zorder[j])
				else:
					p = plt.plot(vals1[:,j], vals2[:,j], c=colours[j], lw=lw, alpha=alpha, zorder=zorder[j])


		plt.ylabel(r"Prediction")
		plt.xlabel(r"True value")
		plt.title(f"Model function vs target for \n{NK[i]} random I.C.s, scanned over {VAR_STR[i]}")
		plt.xlim([-10, 10])
		plt.ylim([-10, 10])
		plt.legend()
		plt.tight_layout()
	
	plt.show()



# THIS NEEDS TO BE PLOTTED RELATIVE TO eg STD DEV OF THAT OUTPUT VARIABLE.
def linear_prediction_errors():
	lin = get_good_linear_fit(enforce_constraints=False)

	for k in range(4):

		ax = plt.gca()

		def error_fn(x):
			return target(x, 0.2)[k] - lin(x)[k]
			# error = np.linalg.norm(pred[k] - actual[k])

		contour_plot(error_fn)#, xi=2, yi=3)

		ax.set_title(r"Error in prediction of " + VAR_STR[k])

		plt.show()



def nonlinear_fit_evaluate():

	model_fn = load_model_function("nonlin_16_12")
	fig, axs = plt.subplots(2, 2)
	axs = axs.flatten()
	axs[0].set_title("True change in " + VAR_STR[1])
	contour_plot(lambda x: target(x)[1], ax=axs[0])
	axs[2].set_title("Predicted change in " + VAR_STR[1])
	contour_plot(lambda x: model_fn(x)[1], ax=axs[2])
	axs[1].set_title("True change in " + VAR_STR[2])
	contour_plot(lambda x: target(x)[2], ax=axs[1])
	axs[3].set_title("Predicted change in " + VAR_STR[2])
	contour_plot(lambda x: model_fn(x)[2], ax=axs[3])
	plt.show()

	model_comparison_scatter(model_fn, [50, 50, 3, 3, 3])







# evaluate single step RMSE over random states
def get_rand_model_single_step_RMSE(model_fn, N=2048, rand_range=P_RANGE5):

	SE = 0.

	for i in range(N):
		x = rand_state5(rand_range)
		y = fast_target(x)
		y_est = model_fn(x)
		SE += np.square(y - y_est)

	return np.sqrt(SE / N)


def get_scan_model_single_step_RMSE(model_fn, si, N=1000, relative=True):

	SE = 0.
	max_abs_ys = np.zeros(4)

	for i, state in variable_scan(idx, N, start=None):
		y = fast_target(state)
		y_est = model_fn(state)

		SE += np.square(y - y_est)
		for i, max_y_i in enumerate(max_abs_ys):
			max_abs_ys[i] = max(max_abs_ys[i], abs(y[i]))

	if relative:
		return np.sqrt(SE / N), max_abs_ys
	else:
		return np.sqrt(SE / N)


def junk_I_think():

	colours = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
	N_noise = 10
	N_trials = 512
	RMSEs = np.zeros((N_noise, 4, 5))
	noise_vals = np.linspace(0., 2., N_noise)
	refs = np.zeros((4, 5))


	for k in range(5):

		# rand_range = np.zeros(5)
		# rand_range[k] = P_RANGE5[k]

		for i, noise_fraction in enumerate(noise_vals):

			N_runs = 1
			for l in range(N_runs):
				a = np.zeros((N_noise, 4))
				b = np.zeros(4)

				n_lin_model = get_good_noisy_linear_fit(noise_fraction, 2**14)
				a[i, :], b = get_scan_model_single_step_RMSE(n_lin_model, k, N_trials, True)
				print(b)
				RMSEs[i, :, k] += a[i,:]
				refs[:,k] += b


			RMSEs[i, :, k] /= N_runs
			refs[:,k] /= N_runs
 
		# s = np.max(RMSEs, axis=0)
		# for i in range(4):
		# 	p = plt.plot(noise_vals, RMSEs[:,i]/s[i], c=colours[i], label=VAR_STR[i])# + "/10")

		# for i in range(4):
		# 	p = plt.plot([noise_vals[0], noise_vals[-1]], [max_abs_ys[i]/s[i], max_abs_ys[i]/s[i]], "--", c=colours[i])


	for j in range(4):

		for k in range(5):
			p = plt.plot(noise_vals, RMSEs[:,j,k], c=colours[k], label=VAR_STR[k] + " scan")
		for k in range(5):
			p = plt.plot([noise_vals[0], noise_vals[-1]], [refs[j,k], refs[j,k]], "--", c=colours[k])



		plt.ylabel(f"RMSE in prediction")
		plt.xlabel(f"Noise scale, as a fraction of parameter range")
		plt.legend()


		plt.title(f"RMSE in " + VAR_STR[j] + f" over single step \n({N_trials} trials)")
		plt.show()







colours = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]


# linear_prediction_errors()
# nonlinear_fit_evaluate()



lin_model = get_good_linear_fit()
# n_lin_model = get_good_noisy_linear_fit(0.3)
lin_model = get_good_noisy_linear_fit(0., N=2**6)
n_lin_model = get_good_noisy_linear_fit(1., N=2**6)


def linear_convergence_w_noise():

	exps = np.linspace(5, 12, 15)#15)
	noise_vals = np.linspace(1., 0., 15)#15)

	cmap = cm.get_cmap("winter", 256)
	cmap = cmap(np.linspace(0, 1, len(noise_vals)))

	typical_values = np.std(target_training_data()[1], axis=0)


	fig, axs = plt.subplots(2, 2)
	axs = axs.flatten()
	for i, exp in enumerate(exps):
		y = np.zeros((len(exps), 4))

		for j, noise_fraction in enumerate(noise_vals):
			print(noise_fraction, int(2**exp))
			n_lin_model = get_good_noisy_linear_fit(noise_fraction, int(2**exp))
			rmse = get_rand_model_single_step_RMSE(n_lin_model, N=512)
			y[j,:] = rmse


		for j in range(4):
			axs[j].plot(noise_vals, y[:, j], c=cmap[i], lw=2, alpha=1)
			axs[j].plot([noise_vals[0], noise_vals[-1]], [typical_values[j]]*2, "k--")
			axs[j].set_ylim(0, 2*typical_values[j])
			if j % 2:
				axs[j].yaxis.tick_right()

	mappable = plt.cm.ScalarMappable(norm=LogNorm(2**exps[0], 2**exps[-1]), cmap="winter")
	divider = make_axes_locatable(axs[1])
	cb = plt.colorbar(mappable, ax=axs)

	plt.plot()

	return
	# this bit is out of favour
	fig, axs = plt.subplots(2, 2)
	axs = axs.flatten()
	for i, noise_fraction in enumerate(noise_vals):
		y = np.zeros((len(exps), 4))

		for j, exp in enumerate(exps):
			print(noise_fraction, int(2**exp))
			n_lin_model = get_good_noisy_linear_fit(noise_fraction, int(2**exp))
			rmse = get_rand_model_single_step_RMSE(n_lin_model, N=512)
			y[j,:] = rmse


		for j in range(4):
			axs[j].plot(2**exps, y[:, j], c=cmap[i], lw=2, alpha=1)
			axs[j].plot([2**exps[0], 2**exps[-1]], [typical_values[j]]*2, "k--")
			axs[j].semilogx()
			axs[j].set_ylim(0, 2*typical_values[j])
			if j % 2:
				axs[j].yaxis.tick_right()

	mappable = plt.cm.ScalarMappable(norm=Normalize(noise_vals[0], noise_vals[-1]), cmap="winter_r")
	plt.colorbar(mappable, ax=axs)
	plt.show()


# linear_convergence_w_noise()

# nonlin_model = get_good_nonlinear_fit()
nonlin_model = get_good_noisy_nonlinear_fit(0.0)
n_nonlin_model = get_good_noisy_nonlinear_fit(.1)

def noisy_target(state):
	return corrupt_single(state, fast_target(state), obs_noise=0.1*P_RANGE5)[1]


while True:
	start_state = rand_state5()
	axs = plot_scan_matrix(fast_target, start_state=start_state)#, fmt="--")
	# axs = plot_scan_matrix(noisy_target, start_state=start_state, fmt="--")

	# plot_scan_matrix(nonlin_model, start_state=start_state, axs_in=axs)
	# plot_scan_matrix(n_nonlin_model, start_state=start_state, axs_in=axs)

	plot_scan_matrix(lin_model, start_state=start_state, axs_in=axs)
	# plot_scan_matrix(n_lin_model, start_state=start_state, axs_in=axs)
	plt.gcf().suptitle("Scans across single variables of target function & linear model")

	plt.show()

# nonlin_model = get_good_nonlinear_fit()

# print("errors:")
# print(get_rand_model_single_step_RMSE(lin_model))


exit()

if __name__ == "__main__":

	f = get_good_nonlinear_fit()

	start_state = rand_state5()
	axs = plot_scan_matrix(fast_target, start_state=start_state)
	plot_scan_matrix(f, start_state=start_state, axs_in=axs)
	plt.show()
