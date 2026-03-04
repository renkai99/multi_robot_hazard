import os
import pickle
import warnings

import numpy as np

from Parameters import Parameters
from Drawer import Drawer
from Matrix import Matrix
from Path_Planner import Path_Planner
from Function_Frame import Function_Frame
from Forward_Greedy_Allocator import Forward_Greedy_Allocator
from Reverse_Greedy_Allocator import Reverse_Greedy_Allocator
from Brute_Force_Allocator import Brute_Force_Allocator

try:
    from shapely.geometry import MultiLineString, Point

    SHAPELY_AVAILABLE = True
except Exception:
    SHAPELY_AVAILABLE = False


# Parameter Functions
def generate_Tau_X(self):
    p_stay = self.p_stay

    for u in self.U_x.U:
        matrix_u = Matrix(self.domain_matrix, np.zeros(tuple([len(e) for e in self.domain_matrix])))
        if u == "0":
            for x in self.X:
                matrix_u.set([x, x], 1)
        else:
            for x in self.X:
                if self.U_x.is_u_in_U_x_(u, x):
                    xx = self.U_x.get_xx_u(x, u)
                    matrix_u.set([x, x], p_stay)
                    matrix_u.set([x, xx], 1 - p_stay)
        self[u] = matrix_u


def sample_Tau_Ys(self, p_f, ys_k_1):
    ys_0 = ~ys_k_1
    ys_1 = ys_k_1

    N_ys = self.X.adj_matrix.T.dot(ys_1.T.astype(int)).T
    D_ys = self.X.adj_diag_matrix.T.dot(ys_1.T.astype(int)).T

    p_cont = 1 - (((1 - p_f) ** N_ys) * ((1 - p_f / np.sqrt(2)) ** D_ys))
    rand = np.random.rand(*ys_1.shape)
    ys_cont = rand <= p_cont

    ys_k = ys_k_1
    ys_k[ys_0] = ys_cont[ys_0]
    return ys_k


class LabGridworld:
    def __init__(self, width, height, x_min=-1.5, x_max=1.5, y_min=-3.0, y_max=3.0):
        self.width = width
        self.height = height
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.dx = (x_max - x_min) / width
        self.dy = (y_max - y_min) / height
        self.gridsize = max(self.dx, self.dy)

    def get_dim(self):
        return self.width, self.height

    def get_center(self, cell):
        x, y = cell
        cx = self.x_min + (x + 0.5) * self.dx
        cy = self.y_min + (y + 0.5) * self.dy
        return (cx, cy)

    def world_to_cell(self, pt):
        x, y = pt
        cx = int((x - self.x_min) / self.dx)
        cy = int((y - self.y_min) / self.dy)
        cx = max(0, min(self.width - 1, cx))
        cy = max(0, min(self.height - 1, cy))
        return (cx, cy)


LAB_OBSTACLE_LINES = [
    [[-1.498, 3.001], [0.001, 3.000]],
    [[1.051, 3.001], [1.494, 3.000]],
    [[1.494, 3.000], [1.493, 0.430]],
    [[1.494, -0.374], [1.497, -2.998]],
    [[0.002, -2.999], [1.497, -2.998]],
    [[-1.498, -2.999], [-1.047, -2.999]],
    [[-1.496, -0.500], [-1.498, -2.999]],
    [[-1.496, 0.750], [-1.495, 0.299]],
    [[-1.498, 2.998], [-1.498, 1.553]],
    [[-0.481, 2.382], [0.879, 1.356]],
    [[-1.498, 1.553], [-0.700, 1.551]],
    [[1.018, 0.429], [1.493, 0.430]],
    [[0.141, 1.040], [-0.269, 0.524]],
    [[-1.496, 0.526], [-0.269, 0.524]],
    [[-0.269, 0.524], [-0.261, -0.008]],
    [[-0.261, -0.008], [0.480, -0.008]],
    [[0.011, -0.008], [0.011, -0.486]],
    [[-1.496, -0.859], [-0.492, -0.860]],
    [[0.922, -0.613], [0.924, -2.093]],
    [[0.260, -1.084], [0.260, -2.093]],
    [[-0.665, -2.094], [0.924, -2.093]],
    [[-0.685, -2.103], [-0.931, -2.414]],
]

LAB_EXIT_LINES = [
    [[-1.498, 1.553], [-1.496, 0.750]],
    [[-1.495, 0.299], [-1.494, -0.500]],
    [[-1.050, -2.999], [0.002, -2.999]],
    [[1.494, -0.374], [1.497, 0.430]],
    [[0.001, 3.001], [1.051, 3.001]],
]


def point_segment_distance(px, py, ax, ay, bx, by):
    abx = bx - ax
    aby = by - ay
    apx = px - ax
    apy = py - ay
    ab2 = abx * abx + aby * aby
    if ab2 == 0:
        return np.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, (apx * abx + apy * aby) / ab2))
    qx = ax + t * abx
    qy = ay + t * aby
    return np.hypot(px - qx, py - qy)


def add_points(gridworld, state, lines, obst=None):
    width, height = gridworld.get_dim()
    threshold = gridworld.gridsize / 2 + 0.03

    if SHAPELY_AVAILABLE:
        linestrings = MultiLineString(lines)
        for x in range(width):
            for y in range(height):
                center = Point(gridworld.get_center((x, y)))
                if center.distance(linestrings) < threshold:
                    if obst is not None and obst[x, y] == 1:
                        continue
                    state[x, y] = 1
        return state

    for x in range(width):
        for y in range(height):
            px, py = gridworld.get_center((x, y))
            min_d = min(
                point_segment_distance(px, py, line[0][0], line[0][1], line[1][0], line[1][1])
                for line in lines
            )
            if min_d < threshold:
                if obst is not None and obst[x, y] == 1:
                    continue
                state[x, y] = 1
    return state


def build_lab_obstacle_map(width=30, height=60):
    gridworld = LabGridworld(width, height)
    obstacles_xy = np.zeros((width, height), dtype=int)
    add_points(gridworld, obstacles_xy, LAB_OBSTACLE_LINES)
    # framework expects map[y, x]
    return gridworld, obstacles_xy.T.astype(int)

def build_exit_cells(gridworld, map_yx):
    width, height = gridworld.get_dim()
    exits_xy = np.zeros((width, height), dtype=int)
    add_points(gridworld, exits_xy, LAB_EXIT_LINES, obst=map_yx.T)
    exit_cells = [(x, y) for x in range(width) for y in range(height) if exits_xy[x, y] == 1]
    if len(exit_cells) == 0:
        # Fallback: use nearest free cell to each exit-segment midpoint.
        midpoints = [((e[0][0] + e[1][0]) / 2.0, (e[0][1] + e[1][1]) / 2.0) for e in LAB_EXIT_LINES]
        return world_points_to_free_cells(gridworld, map_yx, midpoints)
    return sorted(list(set(exit_cells)))


def nearest_free_cell(map_yx, cell):
    width = map_yx.shape[1]
    height = map_yx.shape[0]
    x0, y0 = cell
    x0 = max(0, min(width - 1, x0))
    y0 = max(0, min(height - 1, y0))
    if map_yx[y0, x0] == 0:
        return (x0, y0)

    max_r = max(width, height)
    for r in range(1, max_r + 1):
        x_min = max(0, x0 - r)
        x_max = min(width - 1, x0 + r)
        y_min = max(0, y0 - r)
        y_max = min(height - 1, y0 + r)
        for x in range(x_min, x_max + 1):
            for y in (y_min, y_max):
                if map_yx[y, x] == 0:
                    return (x, y)
        for y in range(y_min + 1, y_max):
            for x in (x_min, x_max):
                if map_yx[y, x] == 0:
                    return (x, y)
    raise ValueError("No free cell found in the map.")


def world_points_to_free_cells(gridworld, map_yx, points):
    out = []
    for pt in points:
        cell = gridworld.world_to_cell(pt)
        out.append(nearest_free_cell(map_yx, cell))
    return out


# Parameters
example_name = "lab_case_study_2r_lowfire"
parameters = Parameters(name=example_name)
open_case_study = False

gridworld, parameters.map = build_lab_obstacle_map(width=16, height=32)

target_world_points = [(-1.05, 2.15), (0.65, -1.55)]
parameters.targets = world_points_to_free_cells(gridworld, parameters.map, target_world_points)
parameters.task_ids = ["i", "ii"]

robot_world_points = [(0.65, 2.75), (-1.35, 1.15)]
parameters.robot_positions = world_points_to_free_cells(gridworld, parameters.map, robot_world_points)
parameters.robot_ids = ["1", "2"]
parameters.robot_linestyles = [(0, ()), (0, (3, 3))]

hazard_world_points = [(0.2, -2.4)]
parameters.y_0 = [[c] for c in world_points_to_free_cells(gridworld, parameters.map, hazard_world_points)]
parameters.hazard_ids = ["a"]
parameters.p_f = [0.05]

parameters.goal = build_exit_cells(gridworld, parameters.map)

parameters.E = 1200
parameters.N = 50
parameters.p_stay = 0

parameters.generate_obsticles()
parameters.generate_Hazards()
parameters.generate_Tasks()
parameters.generate_Robots()

parameters.generate_Tau_X = generate_Tau_X
parameters.sample_Tau_Ys = sample_Tau_Ys

parameters_file = {"Read": open_case_study, "Name": "parameters"}
samples_file = {"Read": open_case_study, "Name": "samples"}
function_frame_file = {"Read": open_case_study, "Name": "function_frame"}
solution_file = {"Read": open_case_study, "Name": "solution"}

### Main ###
warnings.filterwarnings("ignore")

base_dir = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(base_dir, "case_studies", example_name)
if not os.path.exists(path):
    os.makedirs(path)

# Parameters
if parameters_file["Read"]:
    infile = open(os.path.join(path, parameters_file["Name"]), "rb")
    parameters = pickle.load(infile)
    infile.close()
else:
    outfile = open(os.path.join(path, parameters_file["Name"]), "wb")
    pickle.dump(parameters, outfile)
    outfile.close()

parameters.parameters_file = parameters_file
parameters.samples_file = samples_file
parameters.function_frame_file = function_frame_file
parameters.solution_file = solution_file

# Setting up
path_planner = Path_Planner(parameters)
path_planner.set_up(path + os.sep)

Drawer(path_planner).draw_full_example()

# Function frame
if parameters.function_frame_file["Read"]:
    print("...Reading function frame...")
    infile = open(os.path.join(path, parameters.function_frame_file["Name"]), "rb")
    function_frame = pickle.load(infile)
    infile.close()
else:
    function_frame = Function_Frame(parameters, path_planner)
    print("...Saving function frame...")
    outfile = open(os.path.join(path, parameters.function_frame_file["Name"]), "wb")
    pickle.dump(function_frame, outfile)
    outfile.close()

# Forward greedy
allocator_fg = Forward_Greedy_Allocator(function_frame)
if parameters.solution_file["Read"]:
    infile = open(os.path.join(path, parameters.solution_file["Name"] + "_fg"), "rb")
    fg_solution = pickle.load(infile)
    infile.close()
else:
    fg_solution = allocator_fg.solve_problem()
    allocator_fg.postprocess_solution(fg_solution)
    fg_solution.save_solution(os.path.join(path, parameters.solution_file["Name"] + "_fg"))
allocator_fg.show_solution(fg_solution)

# Reverse greedy
allocator_rg = Reverse_Greedy_Allocator(function_frame)
if parameters.solution_file["Read"]:
    infile = open(os.path.join(path, parameters.solution_file["Name"] + "_rg"), "rb")
    rg_solution = pickle.load(infile)
    infile.close()
else:
    rg_solution = allocator_rg.solve_problem()
    allocator_rg.postprocess_solution(rg_solution)
    rg_solution.save_solution(os.path.join(path, parameters.solution_file["Name"] + "_rg"))
allocator_rg.show_solution(rg_solution)

run_brute_force = False
if run_brute_force:
    allocator_bf = Brute_Force_Allocator(function_frame)
    if parameters.solution_file["Read"]:
        infile = open(os.path.join(path, parameters.solution_file["Name"] + "_bf"), "rb")
        bf_solution = pickle.load(infile)
        infile.close()
        infile = open(os.path.join(path, parameters.solution_file["Name"] + "_worst"), "rb")
        worst_solution = pickle.load(infile)
        infile.close()
    else:
        bf_solution, worst_solution = allocator_bf.solve_problem()
        allocator_bf.postprocess_solution(bf_solution)
        allocator_bf.postprocess_solution(worst_solution)
        bf_solution.save_solution(os.path.join(path, parameters.solution_file["Name"] + "_bf"))
        worst_solution.save_solution(os.path.join(path, parameters.solution_file["Name"] + "_worst"))

    allocator_bf.show_solution(bf_solution)
    allocator_bf.show_solution(worst_solution)
