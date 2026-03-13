import argparse
import os
import shutil
import time
import uuid
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Parameters import Parameters
from Matrix import Matrix
from Path_Planner import Path_Planner
from Function_Frame import Function_Frame
from Forward_Greedy_Allocator import Forward_Greedy_Allocator
from Reverse_Greedy_Allocator import Reverse_Greedy_Allocator

try:
    from shapely.geometry import MultiLineString, Point

    SHAPELY_AVAILABLE = True
except Exception:
    SHAPELY_AVAILABLE = False


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

TARGET_WORLD_CANDIDATES = [
    (-1.25, 2.2),
    (-0.2, 2.0),
    (0.9, 2.5),
    (-1.1, 0.9),
    (-0.35, 0.25),
    (0.5, 0.15),
    (0.95, -0.8),
    (0.65, -1.55),
    (-0.2, -1.8),
    (0.1, -2.6),
]


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


# Keep dynamics identical to two_stage_framework_lab.py

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


def build_lab_obstacle_map(width, height):
    gridworld = LabGridworld(width, height)
    obstacles_xy = np.zeros((width, height), dtype=int)
    add_points(gridworld, obstacles_xy, LAB_OBSTACLE_LINES)
    return gridworld, obstacles_xy.T.astype(int)


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
    raise ValueError("No free cell found in map")


def world_points_to_free_cells(gridworld, map_yx, points):
    out = []
    for pt in points:
        cell = gridworld.world_to_cell(pt)
        out.append(nearest_free_cell(map_yx, cell))
    return out


def build_exit_cells(gridworld, map_yx):
    width, height = gridworld.get_dim()
    exits_xy = np.zeros((width, height), dtype=int)
    add_points(gridworld, exits_xy, LAB_EXIT_LINES, obst=map_yx.T)
    exit_cells = [(x, y) for x in range(width) for y in range(height) if exits_xy[x, y] == 1]
    if len(exit_cells) == 0:
        mids = [((e[0][0] + e[1][0]) / 2.0, (e[0][1] + e[1][1]) / 2.0) for e in LAB_EXIT_LINES]
        return world_points_to_free_cells(gridworld, map_yx, mids)
    return sorted(list(set(exit_cells)))


def free_cells_from_map(map_yx):
    h, w = map_yx.shape
    cells = []
    for y in range(h):
        for x in range(w):
            if map_yx[y, x] == 0:
                cells.append((x, y))
    return cells


def choose_targets(gridworld, map_yx, n_tasks, rng):
    candidates = world_points_to_free_cells(gridworld, map_yx, TARGET_WORLD_CANDIDATES)
    unique = []
    for c in candidates:
        if c not in unique:
            unique.append(c)

    free_cells = free_cells_from_map(map_yx)
    rng.shuffle(free_cells)
    for c in free_cells:
        if c not in unique:
            unique.append(c)
        if len(unique) >= n_tasks:
            break

    if len(unique) < n_tasks:
        raise ValueError("Not enough free cells for tasks")
    return unique[:n_tasks]


def sample_positions_from_exits(exit_cells, n, rng):
    if n <= len(exit_cells):
        idx = rng.choice(len(exit_cells), size=n, replace=False)
        return [exit_cells[i] for i in idx]
    idx = rng.choice(len(exit_cells), size=n, replace=True)
    return [exit_cells[i] for i in idx]


def sample_hazard_positions(map_yx, avoid_cells, n, rng):
    free_cells = [c for c in free_cells_from_map(map_yx) if c not in avoid_cells]
    if len(free_cells) == 0:
        raise ValueError("No free cells available for hazards")
    replace = n > len(free_cells)
    idx = rng.choice(len(free_cells), size=n, replace=replace)
    return [free_cells[i] for i in idx]


def make_run_dir(output_dir, prefix):
    run_id = f"{prefix}_{uuid.uuid4().hex[:8]}"
    path = os.path.join(output_dir, "tmp_runs", run_id)
    os.makedirs(path, exist_ok=False)
    return path


def create_parameters(
    case_name,
    map_width,
    map_height,
    n_tasks,
    n_agents,
    n_hazards,
    spread_rate,
    E,
    N,
    rng,
):
    parameters = Parameters(name=case_name)

    gridworld, parameters.map = build_lab_obstacle_map(width=map_width, height=map_height)
    exit_cells = build_exit_cells(gridworld, parameters.map)

    parameters.targets = choose_targets(gridworld, parameters.map, n_tasks=n_tasks, rng=rng)
    parameters.task_ids = [str(i + 1) for i in range(n_tasks)]

    parameters.robot_positions = sample_positions_from_exits(exit_cells, n=n_agents, rng=rng)
    parameters.robot_ids = [str(i + 1) for i in range(n_agents)]
    base_styles = [(0, ()), (0, (3, 3)), (0, (1, 2)), (0, (5, 2)), (0, (2, 2))]
    parameters.robot_linestyles = [base_styles[i % len(base_styles)] for i in range(n_agents)]

    avoid = set(parameters.targets) | set(parameters.robot_positions)
    hazard_cells = sample_hazard_positions(parameters.map, avoid_cells=avoid, n=n_hazards, rng=rng)
    parameters.y_0 = [[c] for c in hazard_cells]
    parameters.hazard_ids = [chr(ord("a") + i) for i in range(n_hazards)]
    parameters.p_f = [spread_rate for _ in range(n_hazards)]

    parameters.goal = exit_cells

    parameters.E = E
    parameters.N = N
    parameters.p_stay = 0

    parameters.generate_obsticles()
    parameters.generate_Hazards()
    parameters.generate_Tasks()
    parameters.generate_Robots()

    parameters.generate_Tau_X = generate_Tau_X
    parameters.sample_Tau_Ys = sample_Tau_Ys

    parameters.parameters_file = {"Read": False, "Name": "parameters"}
    parameters.samples_file = {"Read": False, "Name": "samples"}
    parameters.function_frame_file = {"Read": False, "Name": "function_frame"}
    parameters.solution_file = {"Read": False, "Name": "solution"}
    return parameters


def get_path_visited_cells(path):
    visited = set()
    T = len(path.domain[0])
    for t in range(T):
        i_x = np.where(path.matrix[t, :])[0][0]
        visited.add(path.domain[1][i_x])
    return visited


def compute_task_completion_rate(solution, parameters):
    visited = set()
    for p in solution.path:
        visited |= get_path_visited_cells(p)

    n_completed = sum(1 for task in parameters.tasks if task.target in visited)
    return n_completed / max(1, len(parameters.tasks))


def run_allocator(function_frame, algorithm):
    if algorithm == "forward":
        allocator = Forward_Greedy_Allocator(function_frame)
    elif algorithm == "reverse":
        allocator = Reverse_Greedy_Allocator(function_frame)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    solution = allocator.solve_problem()

    # Skip allocator.postprocess_solution() to avoid expensive alpha/gamma loops.
    t0 = time.time()
    solution = allocator.add_optimal_policies(solution)
    solution = allocator.add_optimal_paths(solution)
    solution = allocator.add_group_objective(solution)
    policy_eval_time = time.time() - t0

    return allocator, solution, policy_eval_time


def run_single_experiment(config, output_dir, rng, algorithm="forward", keep_tmp=False):
    case_name = config["case_name"]
    run_dir = make_run_dir(output_dir, prefix=case_name)

    metrics = {}
    try:
        parameters = create_parameters(
            case_name=case_name,
            map_width=config["map_width"],
            map_height=config["map_height"],
            n_tasks=config["n_tasks"],
            n_agents=config["n_agents"],
            n_hazards=config["n_hazards"],
            spread_rate=config["spread_rate"],
            E=config["E"],
            N=config["N"],
            rng=rng,
        )

        t_setup0 = time.time()
        path_planner = Path_Planner(parameters)
        path_planner.set_up(run_dir + os.sep)
        path_setup_time = time.time() - t_setup0

        function_frame = Function_Frame(parameters, path_planner)
        function_frame_time = float(function_frame.instrument.data.get("calculation_time", np.nan))

        allocator, solution, policy_eval_time = run_allocator(function_frame, algorithm=algorithm)

        safety_rate = solution.objective_value["group"] if isinstance(solution.objective_value, dict) else np.nan
        task_completion_rate = compute_task_completion_rate(solution, parameters)

        alloc_setup_time = solution.time_data.get("setup_time", np.nan)
        alloc_time = solution.time_data.get("calculation_time", np.nan)

        metrics = {
            "ok": True,
            "case_name": case_name,
            "algorithm": solution.algorithm,
            "map_width": config["map_width"],
            "map_height": config["map_height"],
            "map_cells": config["map_width"] * config["map_height"],
            "n_tasks": config["n_tasks"],
            "n_agents": config["n_agents"],
            "n_hazards": config["n_hazards"],
            "spread_rate": config["spread_rate"],
            "E": config["E"],
            "N": config["N"],
            "path_setup_time_s": path_setup_time,
            "function_frame_time_s": function_frame_time,
            "allocator_setup_time_s": alloc_setup_time,
            "allocator_time_s": alloc_time,
            "policy_eval_time_s": policy_eval_time,
            "total_time_s": path_setup_time + function_frame_time + alloc_setup_time + alloc_time + policy_eval_time,
            "safety_rate": safety_rate,
            "task_completion_rate": task_completion_rate,
        }
    except Exception as exc:
        metrics = {
            "ok": False,
            "case_name": case_name,
            "map_width": config.get("map_width"),
            "map_height": config.get("map_height"),
            "n_tasks": config.get("n_tasks"),
            "n_agents": config.get("n_agents"),
            "n_hazards": config.get("n_hazards"),
            "spread_rate": config.get("spread_rate"),
            "E": config.get("E"),
            "N": config.get("N"),
            "error": repr(exc),
        }
    finally:
        if not keep_tmp and os.path.isdir(run_dir):
            shutil.rmtree(run_dir, ignore_errors=True)

    return metrics


def boxplot_metric(df, x_col, y_col, title, xlabel, ylabel, out_file):
    x_vals = sorted(df[x_col].unique())
    data = [df[df[x_col] == x][y_col].dropna().values for x in x_vals]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.boxplot(data, labels=[str(x) for x in x_vals], patch_artist=True)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_file, dpi=180)
    plt.close(fig)


def run_runtime_scaling(output_dir, base_cfg, repeats, rng, algorithm):
    rows = []
    algo_tag = algorithm.lower()

    task_grid = [1, 2, 3, 4]
    for n_tasks in task_grid:
        for rep in range(repeats):
            cfg = dict(base_cfg)
            cfg.update({"n_tasks": n_tasks, "case_name": f"runtime_tasks_t{n_tasks}_r{rep}"})
            rows.append(run_single_experiment(cfg, output_dir, rng, algorithm=algorithm))

    agent_grid = [1, 2, 3, 4]
    for n_agents in agent_grid:
        for rep in range(repeats):
            cfg = dict(base_cfg)
            cfg.update({"n_agents": n_agents, "case_name": f"runtime_agents_a{n_agents}_r{rep}"})
            rows.append(run_single_experiment(cfg, output_dir, rng, algorithm=algorithm))

    size_grid = [(12, 24), (15, 30), (18, 36)]
    for (w, h) in size_grid:
        for rep in range(repeats):
            cfg = dict(base_cfg)
            cfg.update({"map_width": w, "map_height": h, "case_name": f"runtime_size_{w}x{h}_r{rep}"})
            rows.append(run_single_experiment(cfg, output_dir, rng, algorithm=algorithm))

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(output_dir, f"runtime_scaling_raw_{algo_tag}.csv"), index=False)

    ok_df = df[df["ok"] == True].copy()
    if len(ok_df) > 0:
        df_tasks = ok_df[ok_df["case_name"].str.startswith("runtime_tasks_")]
        df_agents = ok_df[ok_df["case_name"].str.startswith("runtime_agents_")]
        df_sizes = ok_df[ok_df["case_name"].str.startswith("runtime_size_")]

        boxplot_metric(
            df_tasks,
            x_col="n_tasks",
            y_col="total_time_s",
            title="Runtime Scaling vs Number of Tasks",
            xlabel="Number of tasks",
            ylabel="Total runtime [s]",
            out_file=os.path.join(output_dir, f"runtime_vs_tasks_boxplot_{algo_tag}.png"),
        )
        boxplot_metric(
            df_agents,
            x_col="n_agents",
            y_col="total_time_s",
            title="Runtime Scaling vs Number of Agents",
            xlabel="Number of agents",
            ylabel="Total runtime [s]",
            out_file=os.path.join(output_dir, f"runtime_vs_agents_boxplot_{algo_tag}.png"),
        )
        boxplot_metric(
            df_sizes,
            x_col="map_cells",
            y_col="total_time_s",
            title="Runtime Scaling vs Map Size",
            xlabel="Number of cells",
            ylabel="Total runtime [s]",
            out_file=os.path.join(output_dir, f"runtime_vs_mapcells_boxplot_{algo_tag}.png"),
        )

    return df


def run_robustness_analysis(output_dir, base_cfg, repeats, rng, algorithm):
    rows_hazard = []
    algo_tag = algorithm.lower()
    hazard_grid = [1, 2, 3, 4]
    for n_hazards in hazard_grid:
        for rep in range(repeats):
            cfg = dict(base_cfg)
            cfg.update({
                "n_hazards": n_hazards,
                "case_name": f"robust_hcount_h{n_hazards}_r{rep}",
            })
            rows_hazard.append(run_single_experiment(cfg, output_dir, rng, algorithm=algorithm))

    df_hazard = pd.DataFrame(rows_hazard)
    df_hazard.to_csv(os.path.join(output_dir, f"robustness_vs_hazard_count_raw_{algo_tag}.csv"), index=False)

    rows_spread = []
    spread_grid = [0.01, 0.02, 0.04, 0.08, 0.10]
    for p_f in spread_grid:
        for rep in range(repeats):
            cfg = dict(base_cfg)
            cfg.update({
                "spread_rate": p_f,
                "case_name": f"robust_spread_p{p_f:.3f}_r{rep}",
            })
            rows_spread.append(run_single_experiment(cfg, output_dir, rng, algorithm=algorithm))

    df_spread = pd.DataFrame(rows_spread)
    df_spread.to_csv(os.path.join(output_dir, f"robustness_vs_spread_raw_{algo_tag}.csv"), index=False)

    ok_h = df_hazard[df_hazard["ok"] == True].copy()
    if len(ok_h) > 0:
        boxplot_metric(
            ok_h,
            x_col="n_hazards",
            y_col="safety_rate",
            title="Safety Rate vs Number of Hazards",
            xlabel="Number of hazards",
            ylabel="Safety rate",
            out_file=os.path.join(output_dir, f"safety_vs_hazard_count_boxplot_{algo_tag}.png"),
        )
        boxplot_metric(
            ok_h,
            x_col="n_hazards",
            y_col="task_completion_rate",
            title="Task Completion vs Number of Hazards",
            xlabel="Number of hazards",
            ylabel="Task completion rate",
            out_file=os.path.join(output_dir, f"task_completion_vs_hazard_count_boxplot_{algo_tag}.png"),
        )

    ok_s = df_spread[df_spread["ok"] == True].copy()
    if len(ok_s) > 0:
        boxplot_metric(
            ok_s,
            x_col="spread_rate",
            y_col="safety_rate",
            title="Safety Rate vs Hazard Spread Rate",
            xlabel="Hazard spread rate p_f",
            ylabel="Safety rate",
            out_file=os.path.join(output_dir, f"safety_vs_spread_boxplot_{algo_tag}.png"),
        )
        boxplot_metric(
            ok_s,
            x_col="spread_rate",
            y_col="task_completion_rate",
            title="Task Completion vs Hazard Spread Rate",
            xlabel="Hazard spread rate p_f",
            ylabel="Task completion rate",
            out_file=os.path.join(output_dir, f"task_completion_vs_spread_boxplot_{algo_tag}.png"),
        )

    return df_hazard, df_spread


def parse_args():
    parser = argparse.ArgumentParser(description="Quantitative analysis on lab environment")
    parser.add_argument("--output-dir", default="case_studies/lab_analysis_outputs")
    parser.add_argument("--mode", choices=["all", "runtime", "robustness"], default="all")
    parser.add_argument("--algorithm", choices=["forward", "reverse"], default="forward")
    parser.add_argument("--runtime-repeats", type=int, default=3)
    parser.add_argument("--robustness-repeats", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--E", type=int, default=800)
    parser.add_argument("--N", type=int, default=45)
    return parser.parse_args()


def main():
    warnings.filterwarnings("ignore")
    args = parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "tmp_runs"), exist_ok=True)

    rng = np.random.default_rng(args.seed)

    base_cfg = {
        "case_name": "lab_analysis_base",
        "map_width": 15,
        "map_height": 30,
        "n_tasks": 2,
        "n_agents": 2,
        "n_hazards": 1,
        "spread_rate": 0.01,
        "E": args.E,
        "N": args.N,
    }

    if args.mode in ["all", "runtime"]:
        df_runtime = run_runtime_scaling(
            output_dir=output_dir,
            base_cfg=base_cfg,
            repeats=args.runtime_repeats,
            rng=rng,
            algorithm=args.algorithm,
        )
        print(f"[runtime] wrote {len(df_runtime)} rows")

    if args.mode in ["all", "robustness"]:
        df_h, df_s = run_robustness_analysis(
            output_dir=output_dir,
            base_cfg=base_cfg,
            repeats=args.robustness_repeats,
            rng=rng,
            algorithm=args.algorithm,
        )
        print(f"[robustness] wrote {len(df_h)} hazard rows and {len(df_s)} spread rows")

    print(f"Outputs saved in: {output_dir}")


if __name__ == "__main__":
    main()
