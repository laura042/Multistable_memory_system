import numpy as np
import matplotlib.pyplot as plt

from ilqr import iLQR
from ilqr.cost import QRCost
from env_dynamics import EnvDynamics

from EQD import Ressort

import argparse

import gym_systmemoire
import gym_systmemoire.envs
import Config_env

import matplotlib
import matplotlib.font_manager as fm
import os


parser = argparse.ArgumentParser()
parser.add_argument("--path_to_save", type=str, default='', help="path to save positions, velocities, output force signal and the associated figures in case of success")
parser.add_argument("--dirname_to_save", type=str, default='', help="dirname to save positions, velocities, output force signal and the associated figures in case of success")
parser.add_argument("--file_name", type=str, default='', help="name of the files to save")
parser.add_argument("--dt", type=float, default=[0.1], help="must be a list")
parser.add_argument("--N", type=int, default=[200], help="must be a list")
parser.add_argument("--env_surname", type=str, default='3M-v0', help="environment to use")
parser.add_argument("--c", default=0.6, help="dissipation value")
parser.add_argument("--transition_gridsearch", default=[''], help="transitions to perform (used in the file names)")
parser.add_argument("--init_state", default=[[]], help="initial state. Not taken into account if random_initial_state is set to True or out_of_eq_init_state is not None.")
parser.add_argument("--target_state", default=[[]], help='target state. Not taken into account if random_target_state is set to True.')
parser.add_argument("--random_initial_state", type=bool, default=True, help="if the initial state is randomly chosen (may be a non equilibrium state)")
parser.add_argument("--random_target_state", type=bool, default=True, help="if the target is randomly chosen")
parser.add_argument("--limit_reset", default=[0.2, 0.1], help="used if random_initial_state is set to True.")
parser.add_argument("--out_of_eq_init_state", default=None, help="must specify each initial position and velocity in a list")
parser.add_argument("--path_for_font", type=str, default=None)
parser.add_argument("--plot_successes", type=bool, default=True, help="plot positions, velocities and output force")
parser.add_argument("--save_successes", type=bool, default=True, help="save positions, velocities and output force")
args = parser.parse_args()

### FONT FOR MATPLOTLIB ###
fe = fm.FontEntry(
    fname=os.path.join(args.path_for_font, './ttf/cmunbx.ttf'),
    name='cmunbx')
fm.fontManager.ttflist.insert(0, fe) # or append is fine
matplotlib.rcParams['font.family'] = fe.name # = 'your custom ttf font name'

plt.rc('font', size=20)
### END FONT FOR MATPLOTLIB ###


def bin_to_pos_general(binstate, nb_pos_eq, eq_pos):
    for idx in range(nb_pos_eq - int(np.trunc(nb_pos_eq / 2))):
        if binstate == idx:
            become = eq_pos[2 * idx]
    return become


def goalbin_to_goalpos(nb_pos_eq, eq_positions, init_state, nb_ressort):
    goalpos = np.zeros_like(init_state, dtype='float')
    goalpos_cum = np.zeros_like(init_state, dtype='float')
    for i in range(nb_ressort):
        goalpos[i] = bin_to_pos_general(init_state[i], nb_pos_eq, eq_positions[i])
    goalpos_cum[0] = goalpos[0]
    for i in range(nb_ressort - 1):
        goalpos_cum[i + 1] = goalpos_cum[i] + goalpos[i + 1]
    goalpos_cum = np.append(goalpos_cum, np.zeros(nb_ressort))
    return goalpos_cum.tolist()

opt_hist = []
dt_hist = []
N_hist = []

for transition_idx in range(len(args.transition_gridsearch)):
    for dt in args.dt:
        for N in args.N:

            masses = Config_env.exp['{}'.format(args.env_surname)]['masse']
            dynamics = EnvDynamics(ressorts=Config_env.exp['{}'.format(args.env_surname)]['system'],
                                   masse=masses,
                                   c_frot=args.c*np.ones(masses.shape[0]),
                                   dt=dt).dynamics


            def on_iteration(iteration_count, xs, us, J_opt, accepted, converged):
                global successes
                J_hist.append(J_opt)
                info = "converged" if converged else ("accepted" if accepted else "failed")
                final_state = dynamics.f(xs[-1], us[-1], 0)
                print("iteration", iteration_count, info, J_opt, final_state)

                #RL condition success
                verif_pos = np.zeros(len(masses))
                verif_vel = np.zeros(len(masses))
                for unit_idx in range(len(masses)):
                    verif_pos[unit_idx] = np.abs(xs[-1, unit_idx] - goal[unit_idx]) < 0.005
                    verif_vel[unit_idx] = np.abs(xs[-1, unit_idx + nb_ressort]) < 0.01
                if converged and np.sum(verif_pos + verif_vel) == 2 * len(masses):
                    print('success')

                    if args.plot_successes or args.save_successes:

                        t = np.arange(N + 1)
                        t_force = np.arange(N)
                        x = []
                        v = []
                        for unit_idx in range(len(masses)):
                            x.append(xs[:, unit_idx])
                            v.append(xs[:, unit_idx + nb_ressort])

                        os.makedirs(os.path.join(args.path_to_save, './{}'.format(args.dirname_to_save)), exist_ok=True)

                        if args.plot_successes:

                            cycle = ['blue', 'orange', 'green', 'purple', 'pink']

                            fig = plt.figure(figsize=(8.5, 6))
                            axes = fig.add_subplot()
                            for unit_idx in range(len(masses)):
                                plt.plot(t, x[unit_idx], '-', linewidth=3, markersize=18, color='tab:{}'.format(cycle[unit_idx]))
                                plt.plot(t, [target_state[unit_idx] for k in range(N + 1)], '--', linewidth=2, color='tab:red')
                            axes.set_xlabel('Number of steps', fontsize=34)
                            axes.set_ylabel('Position', fontsize=34)
                            for tick in axes.xaxis.get_ticklabels():
                                tick.set_weight('bold')
                            for tick in axes.yaxis.get_ticklabels():
                                tick.set_weight('bold')
                            plt.rc('axes', unicode_minus=False)
                            axes.tick_params(axis='both', which='major', labelsize=28, width=1)
                            plt.savefig(os.path.join(args.path_to_save, './{}/positions_transition_{}_dt_{}_N_{}.pdf'.format(args.dirname_to_save, args.transition_gridsearch[transition_idx], dt, N)), format='pdf', bbox_inches='tight')

                            fig = plt.figure(figsize=(8.5, 6))
                            axes = fig.add_subplot()
                            for unit_idx in range(len(masses)):
                                plt.plot(t, v[unit_idx], '-', linewidth=3, markersize=18, color='tab:{}'.format(cycle[unit_idx]))
                            plt.plot(t, [0 for k in range(N + 1)], '--', linewidth=2, color='tab:red')
                            axes.set_xlabel('Number of steps', fontsize=34)
                            axes.set_ylabel('Velocity', fontsize=34)
                            for tick in axes.xaxis.get_ticklabels():
                                tick.set_weight('bold')
                            for tick in axes.yaxis.get_ticklabels():
                                tick.set_weight('bold')
                            plt.rc('axes', unicode_minus=False)
                            axes.tick_params(axis='both', which='major', labelsize=28, width=1)
                            plt.savefig(os.path.join(args.path_to_save, './{}/velocities_transition_{}_dt_{}_N_{}.pdf'.format(args.dirname_to_save, args.transition_gridsearch[transition_idx], dt, N)), format='pdf', bbox_inches='tight')

                            fig = plt.figure(figsize=(8.5, 6))
                            axes = fig.add_subplot()
                            plt.plot(t_force, us[:, 0], '-', linewidth=3, markersize=18, color='tab:blue')
                            axes.set_xlabel('Number of steps', fontsize=34)
                            axes.set_ylabel('Force signal', fontsize=34)
                            for tick in axes.xaxis.get_ticklabels():
                                tick.set_weight('bold')
                            for tick in axes.yaxis.get_ticklabels():
                                tick.set_weight('bold')
                            plt.rc('axes', unicode_minus=False)
                            axes.tick_params(axis='both', which='major', labelsize=28, width=1)
                            plt.savefig(os.path.join(args.path_to_save, './{}/force_transition_{}_dt_{}_N_{}.pdf'.format(args.dirname_to_save, args.transition_gridsearch[transition_idx], dt, N)), format='pdf', bbox_inches='tight')

                        if args.save_successes:
                            positions = np.array(x)
                            velocities = np.array(v)
                            force_signal = np.array(us)

                            with open(os.path.join(args.path_to_save, './{}/pos_{}_c_0_6_transition_{}_dt_{}_N_{}.npy'.format(args.dirname_to_save, args.file_name, args.transition_gridsearch[transition_idx], dt, N)), 'w') as f:
                                np.savetxt(f, positions)
                            with open(os.path.join(args.path_to_save, './{}/vel_{}_c_0_6_transition_{}_dt_{}_N_{}.npy'.format(args.dirname_to_save, args.file_name, args.transition_gridsearch[transition_idx], dt, N)), 'w') as f:
                                np.savetxt(f, velocities)
                            with open(os.path.join(args.path_to_save, './{}/force_{}_c_0_6_transition_{}_dt_{}_N_{}.npy'.format(args.dirname_to_save, args.file_name, args.transition_gridsearch[transition_idx], dt, N)), 'w') as f:
                                np.savetxt(f, force_signal)

                #if no success, find the parameters dt and N that give the best convergence
                if J_opt < 1000.:
                    opt_hist.append(J_opt)
                    dt_hist.append(dt)
                    N_hist.append(N)

            nb_ressort = len(masses)
            eq_positions = np.array([Config_env.exp['{}'.format(args.env_surname)]['system'][k].x_e for k in range(nb_ressort)])
            if args.random_initial_state:
                init_state = []
                for unit_idx in range(nb_ressort):
                    init_state.append(np.random.uniform(low=eq_positions[unit_idx][0]-args.limit_reset[0], high=eq_positions[unit_idx][2]+args.limit_reset[0]))
                for unit_idx in range(nb_ressort):
                    init_state.append(np.random.uniform(low=-args.limit_reset[1], high=args.limit_reset[1]))

            elif args.out_of_eq_init_state is not None:
                init_state = args.out_of_eq_init_state[transition_idx]

            else:
                init_state = goalbin_to_goalpos(nb_pos_eq=3, eq_positions=eq_positions, init_state=args.init_state[transition_idx], nb_ressort=nb_ressort)

            if args.random_target_state:
                target_state = []
                for unit_idx in range(nb_ressort):
                    target_state.append(np.random.choice([0, 1]))

            else:
                target_state = args.target_state[transition_idx]

            target_state = goalbin_to_goalpos(nb_pos_eq=3, eq_positions=eq_positions, init_state=args.target_state[transition_idx], nb_ressort=nb_ressort)
            goal = np.array(target_state)

            print('init state : ', init_state)
            print('target state : ', target_state)

            # Instantenous state cost.
            Q = np.eye(dynamics.state_size)
            R = 0.1 * np.eye(dynamics.action_size)

            # Terminal state cost.
            Q_terminal = 10000 * np.eye(dynamics.state_size)

            cost = QRCost(Q, R, Q_terminal=Q_terminal, x_goal=goal)

            x0 = np.array(init_state)
            us_init = np.random.uniform(-1, 1, (N, dynamics.action_size))
            ilqr = iLQR(dynamics, cost, N)

            J_hist = []

            xs, us = ilqr.fit(x0, us_init, n_iterations=100, on_iteration=on_iteration)

idx = opt_hist.index(np.min(opt_hist))
print('opt : ', opt_hist[idx])
print('N opt : ', N_hist[idx])
print('dt opt : ', dt_hist[idx])




