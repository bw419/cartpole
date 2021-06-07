from globals import *
from utils import *
from cartpole import *
from linear_model import *
from nonlinear_model import *

# USE SOBEL SEQUENCE?
def correlation_plots(function):
	
	NK = 100
	SCAN_N = 10
	corrcoefs = np.zeros((NK, 4, 4))
	gradients = np.zeros((NK, 4, 4))

	for k in range(NK):
		x = rand_state()

		for i in range(4): # input elem

			scan = P_RANGE[i] * np.linspace(-1, 1, SCAN_N)
			vals = np.zeros((SCAN_N, 4))

			x1 = x.copy()
			for l, s in enumerate(scan):
				x1[i] = s
				vals[l, :] += function(x1, 0.2)


			for j in range(4): # output elem

				slope, intercept = np.polyfit(scan, vals[:, j], 1)
				gradients[k,j,i] = slope


				if np.sum(np.abs(vals[:,j]-vals[0, j])) == 0:
					corrcoefs[k,j,i] = np.nan
				else:
					corrcoefs[k,j,i] += np.corrcoef(scan, vals[:, j])[0,1]


	mean_corrcoefs = np.mean(corrcoefs, axis=0)
	mean_gradients = np.mean(gradients, axis=0)
	std_dev_gradients = np.std(gradients, axis=0)


	mean_corrcoefs[np.abs(mean_corrcoefs)<1e-12] = np.nan
	mean_gradients[np.abs(mean_gradients)<1e-12] = np.nan


	show_matrix(mean_corrcoefs, 0, 
		r"$\langle (Z_i, X_j)$ correlation coefficient$\rangle$ for" +  f"\nscans over {NK} random initial states", 
		"Component of X scanned", 
		"Component of Z", div=True)

	show_matrix(np.log10(np.abs(mean_gradients)), 0, 
		r"$\log_{10}\left(\left|\left\langle\frac{dZ_i}{dX_j}\right\rangle\right|\right)$" + f" of best-fit line to\n scans over {NK} random initial states", 
		"Component of X scanned", 
		"Component of Z", div=False)


def causality_plot():

	M = np.zeros((4, 4))
	for i in range(4):
		for j in range(4):
			for k in range(200):
				x = rand_state()
				y1 = target(x, 0.2)
				x[i] += P_RANGE[i]*0.0001
				y2 = target(x, 0.2)

				delta = np.abs(y1[j]-y2[j])
				M[j,i] += delta

	# adjust this to match above

	M /= np.max(M)
	M[np.abs(M)<1e-12] = np.nan

	print(M)
	show_matrix(np.log10(M), 0,
		"Approximate dependencies of system\ntime evolution on state variables", 
		"Parameter varied", 
		"Mean gradient of output ", div=False)


def linear_model_matrix_plot():

	for n in [500, 5000]:
		C = linear_fit(n)
		show_matrix(np.log10(np.abs(C)), 0, 
			r"$\log_{10}(|$Linear model matrix components$|)$" + f"\n with {n} training pairs", axes=False, div=False)


# correlation_plots(single_action)
# correlation_plots(target)
# causality_plot()
# linear_model_matrix_plot()






# nonlin = get_good_nonlinear_fit()
nonlin = load_model_function("nonlin_16_12")
# model_comparison_scatter(lin, 10)




