from globals import *
from utils import *
from scipy.signal import argrelextrema



# CHANGE THESE SO ROLLOUT AND MODEL WORK IN THE SAME WAY?

def plot_rollout(IC, rollout, N=200, remap=False):

	for j in range(len(IC)):
		for t_step in [0.02, 0.2]:
			N_steps = N if t_step==0.02 else int(round(0.1*N))

			T = np.arange(N_steps)*t_step

			states = rollout(IC, N_steps, t_step)

			if remap:
				remapped = remap_angle_v(states[:,2])
				for i in range(1, len(T)):
					if remapped[i] < -(np.pi-1) and remapped[i-1] > (np.pi-1):
						remapped[i] = np.nan
					elif remapped[i] > (np.pi-1) and remapped[i-1] < -(np.pi-1):
						remapped[i] = np.nan
				states[:,2] = remapped

			if t_step == 0.02:
				p = plt.plot(T, states[:,j], label=VAR_STR[j], lw=1.5)

			else:
				plt.plot(T, states[:,j], ls="--", lw=1, c=p[0].get_color(), marker="s", ms=3, mew=1, mec="k", mfc=p[0].get_color())
	

	plt.gcf().set_size_inches((4.8, 4.0))
	plt.title(r"State variable evolution for I.C. " + format_IC(IC))
	plt.xlabel("Time (s)", labelpad=2.0)
	plt.ylabel("State variable value", va="top")
	plt.grid(which="both", alpha=0.2)
	plt.legend(loc="upper right")
	plt.show()


def plot_rollout_comparison(IC, rollout, model_fn, N=50, remap=True, t_step=0.2):

	T = np.arange(N)*t_step
	T3 = np.arange(N*10)*t_step/10
	states1 = rollout(IC, N, t_step)
	states2 = model_fn(IC, N)
	states3 = rollout(IC, N*10, t_step/10)

	for j in range(0,len(IC)):
		if remap:
			for states in [states1, states2, states3]:
				remapped = remap_angle_v(states[:,2] - np.pi) + np.pi
				# for i in range(1, len(T)):
					# if remapped[i] < -(np.pi-1) and remapped[i-1] > (np.pi-1):
						# remapped[i] = np.nan
					# elif remapped[i] > (np.pi-1) and remapped[i-1] < -(np.pi-1):
						# remapped[i] = np.nan
				states[:,2] = remapped

		p=plt.plot(T, states2[:,j], label=VAR_STR[j])
		# plt.plot(T3, states3[:,j], ls="--", lw=0.7, c=p[0].get_color())#, marker="s", ms=3, mew=1, mec="k", mfc=p[0].get_color())
		plt.plot(T, states1[:,j], lw=0, c=p[0].get_color(), marker="s", ms=3, mew=1, mec="k", mfc=p[0].get_color())
	

	plt.gcf().set_size_inches((4.8, 4.0))
	plt.title(r"Modelled trajectory for I.C. " + format_IC(IC))
	plt.xlabel("Time (s)", labelpad=2.0)
	plt.ylabel("Variable value", va="top")
	plt.grid(which="both", alpha=0.2)
	plt.legend(loc="upper right")
	plt.show()