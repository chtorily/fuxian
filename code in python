import numpy as np
import random
import json
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional, Set
import time
from scipy.special import erfinv

# 参数设置
AIRSPACE_SIZE = 400  # 空域单元大小（米）
GRID_SIZE = 20  # 网格大小（米）
NUM_ENTRANCE_PROTECTION_LAYERS = 3  # 入口保护网格层数
SPEED_RANGE_PLANNED = [15, 20]  # 计划速度范围（米/秒）
SPEED_RANGE_PERFORMANCE = [0, 27.8]  # 性能速度范围（米/秒）
SIMULATION_DURATION = 10 * 60  # 模拟时长（秒）
TIME_STEP_LENGTH = 2  # 时间步长（秒）
POSITIONING_ERROR_RADIUS = 40  # 定位误差半径（米）
SAFETY_THRESHOLD = 0.0230  # 安全阈值
OCCUPANCY_RATE_THRESHOLD = 0.0001  # 占用率识别阈值
MAX_CTA_POSTPONEMENT = 5  # 最大CTA推迟次数


# 计算标准差 sigma
SIGMA = POSITIONING_ERROR_RADIUS / (np.sqrt(2) * erfinv(0.95))


class UAVTrajectoryPlanner:
    def __init__(self):
        self.traffic_densities = [10, 20, 30, 40, 50, 60]
        self.scenes_per_density = 100
        self.occupancy_rate_map = self.build_occupancy_rate_map(
            GRID_SIZE, SIGMA, 100, OCCUPANCY_RATE_THRESHOLD
        )

    def generate_scene(self, density: int) -> List[Dict]:
        num_uavs = int(density * SIMULATION_DURATION / 60)
        scene = []

        for _ in range(num_uavs):
            entry_point = self.get_random_grid_point()
            exit_point = self.get_random_grid_point()

            while np.array_equal(entry_point, exit_point):
                exit_point = self.get_random_grid_point()

            entry_time = random.uniform(0, SIMULATION_DURATION)
            distance = np.linalg.norm(np.array(exit_point) - np.array(entry_point))

            min_speed, max_speed = SPEED_RANGE_PLANNED
            min_time = distance / max_speed
            max_time = distance / min_speed
            exit_time = entry_time + random.uniform(min_time, max_time)

            if exit_time <= entry_time:
                exit_time = entry_time + max_time

            scene.append({
                "entry_point": tuple(entry_point),
                "exit_point": tuple(exit_point),
                "entry_time": float(entry_time),
                "exit_time": float(exit_time)
            })

        return scene

    def get_random_grid_point(self) -> Tuple[int, int]:
        x = random.randint(0, AIRSPACE_SIZE // GRID_SIZE - 1) * GRID_SIZE + GRID_SIZE // 2
        y = random.randint(0, AIRSPACE_SIZE // GRID_SIZE - 1) * GRID_SIZE + GRID_SIZE // 2
        return (x, y)

    def calculate_occupancy_rate(self,
                                 planned_point: Tuple[float, float],
                                 grid_center: Tuple[float, float]) -> float:
        Ax, Ay = planned_point
        ix, iy = grid_center

        grid_x_start = ix - GRID_SIZE / 2
        grid_x_end = ix + GRID_SIZE / 2
        grid_y_start = iy - GRID_SIZE / 2
        grid_y_end = iy + GRID_SIZE / 2

        x = np.linspace(grid_x_start, grid_x_end, 100)
        y = np.linspace(grid_y_start, grid_y_end, 100)
        dx = x[1] - x[0]
        dy = y[1] - y[0]

        X, Y = np.meshgrid(x, y)
        Z = (1 / (2 * np.pi * SIGMA ** 2)) * np.exp(
            -((X - Ax) ** 2 + (Y - Ay) ** 2) / (2 * SIGMA ** 2)
        )

        return float(np.sum(Z) * dx * dy)

    def build_occupancy_rate_map(self,
                                 grid_size: int,
                                 sigma: float,
                                 max_distance: float,
                                 threshold: float) -> Dict:
        occupancy_rate_map = {}
        max_grid_offset = int(max_distance / grid_size) + 1

        for dx in range(-max_grid_offset, max_grid_offset + 1):
            for dy in range(-max_grid_offset, max_grid_offset + 1):
                grid_center = (dx * grid_size, dy * grid_size)
                occupancy_rate = self.calculate_occupancy_rate((0, 0), grid_center)
                if occupancy_rate >= threshold:
                    occupancy_rate_map[(dx, dy)] = occupancy_rate

        return occupancy_rate_map

    def detect_conflicts(self,
                         uav_list: List[Dict],
                         safety_threshold: float) -> Tuple[bool, Dict]:
        grid_occupancy = {}

        for uav in uav_list:
            trajectory = uav['planned_trajectory']
            entry_time = int(trajectory['entry_time'])
            exit_time = int(trajectory['exit_time'])

            if entry_time >= exit_time:
                continue

            start_pos = np.array(trajectory['entry_point'])
            end_pos = np.array(trajectory['exit_point'])

            for tau in range(entry_time, exit_time):
                t = (tau - entry_time) / (exit_time - entry_time)
                current_pos = start_pos + t * (end_pos - start_pos)
                grid_center = tuple(GRID_SIZE * np.round(current_pos / GRID_SIZE))

                occupancy_rate = self.calculate_occupancy_rate(tuple(current_pos), grid_center)
                grid_key = (*grid_center, tau)

                if grid_key not in grid_occupancy:
                    grid_occupancy[grid_key] = {
                        'total_occupancy': 0.0,
                        'uav_list': []
                    }

                grid_occupancy[grid_key]['total_occupancy'] += occupancy_rate
                grid_occupancy[grid_key]['uav_list'].append(uav['id'])

                if grid_occupancy[grid_key]['total_occupancy'] > safety_threshold:
                    return True, grid_occupancy

        return False, grid_occupancy

    def generate_reachable_grids(self, current_grid: Tuple[int, int], current_time: int, obstacles: Dict) -> Set[Tuple[int, int]]:
        reachable_grids = set()

        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0),
                       (1, 1), (-1, 1), (1, -1), (-1, -1)]:
            neighbor = (
                current_grid[0] + dx * GRID_SIZE,
                current_grid[1] + dy * GRID_SIZE
            )

            if (0 <= neighbor[0] < AIRSPACE_SIZE and
                    0 <= neighbor[1] < AIRSPACE_SIZE):
                grid_key = (*neighbor, current_time)
                if grid_key not in obstacles or obstacles[grid_key]['total_occupancy'] <= SAFETY_THRESHOLD:
                    reachable_grids.add(neighbor)

        return reachable_grids

    def build_weighted_digraph(self, start: Tuple[float, float], goal: Tuple[float, float], obstacles: Dict, entry_time: int, exit_time: int) -> Dict:
        weighted_digraph = {}
        open_set = {start}
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}

        while open_set:
            current = min(open_set, key=lambda x: f_score.get(x, float('inf')))

            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                path.reverse()
                return path

            open_set.remove(current)

            current_time = entry_time + int(np.linalg.norm(np.array(current) - np.array(start)) / GRID_SIZE)
            if current_time >= exit_time:
                continue

            reachable_grids = self.generate_reachable_grids(current, current_time, obstacles)

            for neighbor in reachable_grids:
                tentative_g_score = g_score[current] + GRID_SIZE

                if (neighbor not in g_score or
                        tentative_g_score < g_score[neighbor]):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = g_score[neighbor] + self.heuristic(neighbor, goal)
                    if neighbor not in open_set:
                        open_set.add(neighbor)

        return None

    def heuristic(self, a: Tuple[float, float], b: Tuple[float, float]) -> float:
        return np.linalg.norm(np.array(a) - np.array(b))

    def find_conflict_free_trajectory(self,
                                      uav: Dict,
                                      obstacles: Dict) -> Optional[List[Tuple[float, float]]]:
        original_entry_time = int(uav['entry_time'])
        original_exit_time = int(uav['exit_time'])
        entry_point = tuple(uav['entry_point'])
        exit_point = tuple(uav['exit_point'])

        for postpone_count in range(MAX_CTA_POSTPONEMENT + 1):  # 包含原始时间窗口的尝试
            current_entry_time = original_entry_time + postpone_count * TIME_STEP_LENGTH
            current_exit_time = original_exit_time + postpone_count * TIME_STEP_LENGTH

            # 检查延迟后的时间窗口是否超出仿真时长
            if current_exit_time > SIMULATION_DURATION:
                return None

            # 尝试在当前时间窗口内规划轨迹
            path = self.build_weighted_digraph(
                entry_point,
                exit_point,
                obstacles,
                current_entry_time,
                current_exit_time
            )

            if path:
                # 更新UAV的时间信息
                uav['entry_time'] = float(current_entry_time)
                uav['exit_time'] = float(current_exit_time)
                return path

        return None

    def calculate_delay_time(self,
                             original_trajectory: Dict,
                             new_trajectory: List[Tuple[float, float]]) -> float:
        """计算延迟时间，考虑实际延迟的时间步长"""
        original_duration = original_trajectory['exit_time'] - original_trajectory['entry_time']
        actual_duration = (len(new_trajectory) - 1) * TIME_STEP_LENGTH
        delay = (actual_duration - original_duration) if actual_duration > original_duration else 0
        return delay

    def build_weighted_digraph(self, start: Tuple[float, float], goal: Tuple[float, float],
                               obstacles: Dict, entry_time: int, exit_time: int) -> Dict:
        weighted_digraph = {}
        open_set = {start}
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}

        # 增加时间窗口检查
        max_allowed_time = min(exit_time, SIMULATION_DURATION)

        while open_set:
            current = min(open_set, key=lambda x: f_score.get(x, float('inf')))

            if current == goal:
                # 检查是否在允许的时间范围内完成
                path = []
                temp = current
                while temp in came_from:
                    path.append(temp)
                    temp = came_from[temp]
                path.append(start)
                path.reverse()

                path_duration = (len(path) - 1) * TIME_STEP_LENGTH
                if entry_time + path_duration <= max_allowed_time:
                    return path
                return None

            open_set.remove(current)

            current_time = entry_time + int(g_score[current] / GRID_SIZE * TIME_STEP_LENGTH)
            if current_time >= max_allowed_time:
                continue

            reachable_grids = self.generate_reachable_grids(current, current_time, obstacles)

            for neighbor in reachable_grids:
                # 考虑时间步长的移动代价
                movement_cost = GRID_SIZE * TIME_STEP_LENGTH
                tentative_g_score = g_score[current] + movement_cost

                if (neighbor not in g_score or
                        tentative_g_score < g_score[neighbor]):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = (g_score[neighbor] +
                                         self.heuristic(neighbor, goal))
                    if neighbor not in open_set:
                        open_set.add(neighbor)

        return None

    def generate_all_scenes(self) -> Dict:
        all_scenes = {}
        for density in self.traffic_densities:
            scenes = [self.generate_scene(density)
                      for _ in range(self.scenes_per_density)]
            all_scenes[str(density)] = scenes
        return all_scenes

    def run_simulation(self, scenes: Dict) -> Dict:
        results = {}

        for density in self.traffic_densities:
            density_results = []

            for scene_idx, scene in enumerate(scenes[str(density)]):
                scene_start_time = time.time()
                uavs = []

                for i, uav_data in enumerate(scene, 1):
                    uav = {
                        'id': i,
                        'entry_point': uav_data['entry_point'],
                        'exit_point': uav_data['exit_point'],
                        'entry_time': uav_data['entry_time'],
                        'exit_time': uav_data['exit_time'],
                        'planned_trajectory': uav_data.copy(),
                        'computing_time': 0,
                        'extra_energy': 0,
                        'initial_energy': self.calculate_initial_energy(uav_data),
                        'delay_time': 0
                    }
                    uavs.append(uav)

                has_conflicts, occupancy_map = self.detect_conflicts(uavs, SAFETY_THRESHOLD)

                if has_conflicts:
                    for uav in uavs:
                        start_time = time.time()
                        new_trajectory = self.find_conflict_free_trajectory(uav, occupancy_map)
                        uav['computing_time'] = time.time() - start_time

                        if new_trajectory:
                            uav['optimal_trajectory'] = new_trajectory
                            uav['extra_energy'] = self.calculate_extra_energy(
                                uav['planned_trajectory'], new_trajectory
                            )
                            uav['delay_time'] = self.calculate_delay_time(
                                uav['planned_trajectory'], new_trajectory
                            )
                        else:
                            uav['optimal_trajectory'] = None
                            uav['extra_energy'] = float('inf')
                            uav['delay_time'] = float('inf')

                scene_results = {
                    'duration': time.time() - scene_start_time,
                    'uavs': uavs
                }
                density_results.append(scene_results)

            results[str(density)] = density_results

        return results

    def calculate_initial_energy(self, trajectory: Dict) -> float:
        distance = np.linalg.norm(
            np.array(trajectory['exit_point']) - np.array(trajectory['entry_point'])
        )
        return distance * 1.0

    def calculate_extra_energy(self,
                               original_trajectory: Dict,
                               new_trajectory: List[Tuple[float, float]]) -> float:
        original_distance = np.linalg.norm(
            np.array(original_trajectory['exit_point']) - np.array(original_trajectory['entry_point'])
        )

        new_distance = sum(
            np.linalg.norm(np.array(new_trajectory[i + 1]) - np.array(new_trajectory[i]))
            for i in range(len(new_trajectory) - 1)
        )

        return max(0, new_distance - original_distance)

    def calculate_delay_time(self,
                             original_trajectory: Dict,
                             new_trajectory: List[Tuple[float, float]]) -> float:
        original_time = (original_trajectory['exit_time'] - original_trajectory['entry_time'])
        new_time = len(new_trajectory) * TIME_STEP_LENGTH
        return max(0, new_time - original_time)


def main():
    planner = UAVTrajectoryPlanner()

    print("Generating scenes...")
    all_scenes = planner.generate_all_scenes()

    with open("generated_scenes.json", "w") as f:
        json.dump(all_scenes, f)

    print("Running simulation...")
    results = planner.run_simulation(all_scenes)

    with open("simulation_results.json", "w") as f:
        json.dump(results, f)

    print("Evaluating performance...")
    evaluate_performance("simulation_results.json")


def evaluate_performance(results_file: str):
    with open(results_file, "r") as f:
        results = json.load(f)

    metrics = {
        "conflict_resolution_success_rate": [],
        "average_computing_time": [],
        "average_extra_energy_consumption_rate": [],
        "average_delay_time_rate": []
    }

    traffic_densities = [10, 20, 30, 40, 50, 60]

    for density in traffic_densities:
        density_results = results[str(density)]

        success_rates = []
        computing_times = []
        extra_energy_rates = []
        delay_time_rates = []

        for scene_result in density_results:
            num_uavs = len(scene_result['uavs'])
            if num_uavs == 0:
                continue

            num_resolved = sum(
                1 for uav in scene_result['uavs']
                if 'optimal_trajectory' in uav and uav['optimal_trajectory'] is not None
            )
            success_rates.append(num_resolved / num_uavs)

            avg_computing_time = np.mean([
                uav['computing_time']
                for uav in scene_result['uavs']
            ])
            computing_times.append(avg_computing_time)

            total_initial_energy = sum(
                uav['initial_energy']
                for uav in scene_result['uavs']
            )
            if total_initial_energy > 0:
                total_extra_energy = sum(
                    uav['extra_energy']
                    for uav in scene_result['uavs']
                )
                extra_energy_rates.append(total_extra_energy / total_initial_energy)

            avg_delay_time = np.mean([
                uav['delay_time']
                for uav in scene_result['uavs']
            ])
            delay_time_rates.append(avg_delay_time / SIMULATION_DURATION)

        metrics["conflict_resolution_success_rate"].append(np.mean(success_rates))
        metrics["average_computing_time"].append(np.mean(computing_times))
        metrics["average_extra_energy_consumption_rate"].append(np.mean(extra_energy_rates))
        metrics["average_delay_time_rate"].append(np.mean(delay_time_rates))

    plot_performance(metrics, traffic_densities)


def plot_performance(metrics: Dict, traffic_densities: List[int]):
    try:
        fig, axs = plt.subplots(2, 2, figsize=(15, 12))

        plot_configs = [
            {
                'metric': 'conflict_resolution_success_rate',
                'title': 'Conflict Resolution Success Rate',
                'ylabel': 'Success Rate',
                'color': 'blue',
                'marker': 'o'
            },
            {
                'metric': 'average_computing_time',
                'title': 'Average Computing Time',
                'ylabel': 'Time (seconds)',
                'color': 'green',
                'marker': 's'
            },
            {
                'metric': 'average_extra_energy_consumption_rate',
                'title': 'Extra Energy Consumption Rate',
                'ylabel': 'Rate',
                'color': 'red',
                'marker': '^'
            },
            {
                'metric': 'average_delay_time_rate',
                'title': 'Average Delay Time Rate',
                'ylabel': 'Rate',
                'color': 'purple',
                'marker': 'd'
            }
        ]

        for idx, config in enumerate(plot_configs):
            row = idx // 2
            col = idx % 2
            ax = axs[row, col]

            ax.plot(
                traffic_densities,
                metrics[config['metric']],
                color=config['color'],
                marker=config['marker'],
                linewidth=2,
                markersize=8,
                label=config['title']
            )

            ax.set_title(config['title'], fontsize=12)
            ax.set_xlabel('Traffic Density (UAVs/min)', fontsize=10)
            ax.set_ylabel(config['ylabel'], fontsize=10)
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.tick_params(axis='both', labelsize=9)

            for x, y in zip(traffic_densities, metrics[config['metric']]):
                ax.annotate(
                    f'{y:.3f}',
                    (x, y),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha='center',
                    fontsize=8
                )

            ax.legend()

        plt.tight_layout()
        save_path = 'performance_evaluation.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()

    except Exception as e:
        print(f"绘图过程中出现错误: {str(e)}")
        print(f"错误类型: {type(e)}")
        import traceback
        print("详细错误信息:")
        print(traceback.format_exc())


def evaluate_performance(results_file: str):
    try:
        with open(results_file, "r") as f:
            results = json.load(f)

        metrics = {
            "conflict_resolution_success_rate": [],
            "average_computing_time": [],
            "average_extra_energy_consumption_rate": [],
            "average_delay_time_rate": []
        }

        traffic_densities = [10, 20, 30, 40, 50, 60]

        for density in traffic_densities:
            density_results = results[str(density)]

            success_rates = []
            computing_times = []
            extra_energy_rates = []
            delay_time_rates = []

            for scene_result in density_results:
                num_uavs = len(scene_result['uavs'])
                if num_uavs == 0:
                    continue

                num_resolved = sum(
                    1 for uav in scene_result['uavs']
                    if 'optimal_trajectory' in uav and uav['optimal_trajectory'] is not None
                )
                success_rates.append(num_resolved / num_uavs)

                avg_computing_time = np.mean([
                    uav['computing_time']
                    for uav in scene_result['uavs']
                ])
                computing_times.append(avg_computing_time)

                total_initial_energy = sum(
                    uav['initial_energy']
                    for uav in scene_result['uavs']
                )
                if total_initial_energy > 0:
                    total_extra_energy = sum(
                        uav['extra_energy']
                        for uav in scene_result['uavs']
                    )
                    extra_energy_rates.append(total_extra_energy / total_initial_energy)

                avg_delay_time = np.mean([
                    uav['delay_time']
                    for uav in scene_result['uavs']
                ])
                delay_time_rates.append(avg_delay_time / SIMULATION_DURATION)

            metrics["conflict_resolution_success_rate"].append(np.mean(success_rates))
            metrics["average_computing_time"].append(np.mean(computing_times))
            metrics["average_extra_energy_consumption_rate"].append(np.mean(extra_energy_rates))
            metrics["average_delay_time_rate"].append(np.mean(delay_time_rates))

        plot_performance(metrics, traffic_densities)

    except Exception as e:
        print(f"性能评估过程中出现错误: {str(e)}")
        print(f"错误类型: {type(e)}")
        import traceback
        print("详细错误信息:")
        print(traceback.format_exc())


def main():
    try:
        planner = UAVTrajectoryPlanner()

        print("开始生成场景...")
        all_scenes = planner.generate_all_scenes()
        print("场景生成完成")

        scene_file = "generated_scenes.json"
        print(f"保存场景到文件: {scene_file}")
        with open(scene_file, "w") as f:
            json.dump(all_scenes, f)
        print("场景保存完成")

        print("开始运行仿真...")
        results = planner.run_simulation(all_scenes)
        print("仿真完成")

        results_file = "simulation_results.json"
        print(f"保存仿真结果到文件: {results_file}")
        with open(results_file, "w") as f:
            json.dump(results, f)
        print("结果保存完成")

        print("开始性能评估...")
        evaluate_performance(results_file)
        print("性能评估完成")

        print("所有流程执行完成!")

    except Exception as e:
        print(f"程序执行过程中出现错误: {str(e)}")
        print(f"错误类型: {type(e)}")
        import traceback
        print("详细错误信息:")
        print(traceback.format_exc())


if __name__ == "__main__":
    main()
