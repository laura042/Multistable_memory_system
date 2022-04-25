import argparse

import numpy as np

from Functions import Force_analysis

parser = argparse.ArgumentParser()
parser.add_argument("--path_res", type=str, default=None, help="path for pos/vel/force results")
parser.add_argument("--transition_name", type=str, default=None, help="transition name for saving")
parser.add_argument("--c_gridsearch", type=np.array, default=np.array([0., 1., 2., 3., 4., 5., 6., 7., 8.]), help="dissipation gridsearch")
parser.add_argument("--path_for_fig", type=str, default=None, help="path for saving figure")
parser.add_argument("--fig_name", type=str, default=None, help="figure name")
args = parser.parse_args()


force_analysis = Force_analysis(path_res=args.path_res,
                                nb_of_tests=1,
                                c_gridsearch=args.c_gridsearch,
                                transition_name=args.tranistion_name,
                                path_for_fig=args.path_for_fig,
                                fig_name=args.fig_name)

if __name__ == '__main__':
    force_analysis
