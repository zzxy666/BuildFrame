import numpy as np
import laspy
from scipy.spatial import cKDTree
from collections import deque
import time
import os
import open3d as o3d

class GroundFilter:
    def __init__(self, pixel_size=1.5, z_reserve=4.0, z_differ=0.1, 
                 fine_z_differ=0.02, k_neighbors=5):
        self.pixel_size = pixel_size
        self.z_reserve = z_reserve
        self.z_differ = z_differ
        self.fine_z_differ = fine_z_differ
        self.k_neighbors = k_neighbors
        
        # 将在process方法中初始化的属性
        self.min_x = self.min_y = self.max_x = self.max_y = None
        self.rows = self.cols = None
        self.min_map = None
        self.base_map = None
        self.real_map = None
        self.grid_point_indices = None
        self.ground_grid = None
        
    def process(self, points):
        """处理点云，返回地面点标签"""
        # 1. 计算点云范围
        self._calculate_range(points)
        
        # 2. 创建高程栅格
        self._create_elevation_grids(points)

        # 3. 填充空白栅格
        self._fill_empty_cells_with_nearest()

        # 4. 区域生长
        self._region_growing()
        
        # 5. 非地面点插值处理
        ground_labels = self._refine_processing(points)
        
        return ground_labels

    def _calculate_range(self, points):
        """计算点云范围"""
        self.min_x, self.min_y = np.min(points[:, 0]), np.min(points[:, 1])
        self.max_x, self.max_y = np.max(points[:, 0]), np.max(points[:, 1])
        
        # 计算栅格行列数
        self.cols = int(np.ceil((self.max_x - self.min_x) / self.pixel_size))
        self.rows = int(np.ceil((self.max_y - self.min_y) / self.pixel_size))
        
    def _create_elevation_grids(self, points):
        """创建高程栅格"""
        # 初始化栅格
        self.min_map = np.full((self.rows, self.cols), np.inf)
        self.base_map = np.full((self.rows, self.cols), -np.inf)
        self.real_map = np.zeros((self.rows, self.cols), dtype=bool)
        self.grid_point_indices = [[[] for _ in range(self.cols)] for _ in range(self.rows)]
        
        # 计算点所在的栅格索引
        col_indices = np.clip(((points[:, 0] - self.min_x) / self.pixel_size).astype(int), 0, self.cols-1)
        row_indices = np.clip(((points[:, 1] - self.min_y) / self.pixel_size).astype(int), 0, self.rows-1)
        
        # 构建最小高程图
        for i in range(len(points)):
            r, c = row_indices[i], col_indices[i]
            if points[i, 2] < self.min_map[r, c]:
                self.min_map[r, c] = points[i, 2]
        
        # 构建基础高程图和候选地面点
        for i in range(len(points)):
            r, c = row_indices[i], col_indices[i]
            if points[i, 2] < self.min_map[r, c] + self.z_reserve:
                if points[i, 2] > self.base_map[r, c]:
                    self.base_map[r, c] = points[i, 2]
                self.real_map[r, c] = True
                self.grid_point_indices[r][c].append(i)  # 每个栅格存储点的索引
    
    
    def _region_growing(self):
        """区域生长算法"""
        # 标记已处理栅格
        processed = np.zeros((self.rows, self.cols), dtype=bool)
        self.ground_grid = np.zeros((self.rows, self.cols), dtype=bool)
        max_region = []
        
        # 8邻域方向
        directions = [(-1,-1), (-1,0), (-1,1),
                     (0,-1),           (0,1),
                     (1,-1),  (1,0),   (1,1)]
        
        for i in range(self.rows):
            for j in range(self.cols):
                if processed[i, j] or not self.real_map[i, j]:
                    continue
                
                current_region = []
                queue = deque([(i, j)])
                processed[i, j] = True
                
                while queue:
                    r, c = queue.popleft()
                    current_region.append((r, c))
                    
                    for dr, dc in directions:
                        nr, nc = r + dr, c + dc
                        if (0 <= nr < self.rows and 0 <= nc < self.cols and
                            not processed[nr, nc] and self.real_map[nr, nc]):
                            if abs(self.base_map[nr, nc] - self.base_map[r, c]) <= self.z_differ:
                                queue.append((nr, nc))
                                processed[nr, nc] = True
                
                if len(current_region) > len(max_region):
                    max_region = current_region
        
        # 标记最大连通区域为地面
        for (r, c) in max_region:
            self.ground_grid[r, c] = True
    
    def _refine_processing(self, points):
        """非地面点插值处理"""
        ground_labels = np.ones(len(points), dtype=int)  # 1为非地面点
        
        # 初始标记地面点（位于地面栅格中的候选点）
        for i in range(self.rows):
            for j in range(self.cols):
                if self.ground_grid[i, j]:
                    for idx in self.grid_point_indices[i][j]:
                        ground_labels[idx] = 2  # 地面点标记为2
        
        # 收集地面栅格中心点和对应高程
        ground_points = []
        for i in range(self.rows):
            for j in range(self.cols):
                if self.ground_grid[i, j]:
                    center_x = self.min_x + (j + 0.5) * self.pixel_size
                    center_y = self.min_y + (i + 0.5) * self.pixel_size
                    ground_points.append([center_x, center_y, self.base_map[i, j]])
        
        if not ground_points:
            return ground_labels
        
        ground_points = np.array(ground_points)
        # 构建KDTree进行近邻搜索
        tree = cKDTree(ground_points[:, :2])
        
        # 对非地面栅格进行插值处理
        for i in range(self.rows):
            for j in range(self.cols):
                if not self.ground_grid[i, j] and self.real_map[i, j]:
                    center_x = self.min_x + (j + 0.5) * self.pixel_size
                    center_y = self.min_y + (i + 0.5) * self.pixel_size
                    
                    # 寻找最近的k个地面栅格
                    distances, indices = tree.query([center_x, center_y], k=self.k_neighbors)
                    
                    # 反距离权重插值
                    weights = 1.0 / (distances + 1e-6)
                    weights /= np.sum(weights)
                    interpolated_z = np.sum(ground_points[indices, 2] * weights)
                    
                    # 重新分类非地面栅格中的点
                    for idx in self.grid_point_indices[i][j]:
                        if points[idx, 2] <= interpolated_z + self.fine_z_differ:
                            ground_labels[idx] = 2
        
        return ground_labels

    def _fill_empty_cells_with_nearest(self):
        """使用最邻近插值填补空像元"""
        # 查找所有空像元和非空像元
        empty_rows, empty_cols = np.where(~self.real_map)
        non_empty_rows, non_empty_cols = np.where(self.real_map)
        print(f"空像元数量: {len(empty_rows)}")
        print(f"非空像元数量: {len(non_empty_rows)}")
        if len(empty_rows) == 0 or len(non_empty_rows) == 0:
            return
        
        # 获取非空像元的位置和对应的高程
        non_empty_positions = np.column_stack((non_empty_rows, non_empty_cols))
        non_empty_min_values = self.min_map[self.real_map]
        
        # 使用KDTree快速查找最近邻
        tree = cKDTree(non_empty_positions)
        
        # 对于每个空像元，找到最近的非空像元
        _, nearest_indices = tree.query(np.column_stack((empty_rows, empty_cols)), k=1)
        
        # 填补空像元
        for idx, (r, c) in enumerate(zip(empty_rows, empty_cols)):
            nearest_idx = nearest_indices[idx]
            nearest_value = non_empty_min_values[nearest_idx]
            
            self.min_map[r, c] = nearest_value
            self.base_map[r, c] = nearest_value
            self.real_map[r, c] = True
            # 添加虚拟点索引(-1表示虚拟点)
            self.grid_point_indices[r][c].append(-1)

def filter_and_visualize(input_path, output_dir=None):
    """处理LAS文件并可视化结果"""
    # 读取LAS文件
    las = laspy.read(input_path)
    points = np.vstack((las.x, las.y, las.z)).T
    
    # 初始化地面滤波器
    ground_filter = GroundFilter(
        pixel_size=1.5, 
        z_reserve=0.5,
        z_differ=0.1,
        fine_z_differ=0.02,
        k_neighbors=5
    )
    
    # 处理点云
    start_time = time.time()
    ground_labels = ground_filter.process(points)
    end_time = time.time()
    
    # 打印处理结果
    print(f"处理完成! 耗时: {end_time - start_time:.2f}秒")
    print(f"总点数: {len(points)}")
    print(f"地面点数量: {np.sum(ground_labels == 2)} ({np.sum(ground_labels == 2)/len(points)*100:.2f}%)")
    print(f"非地面点数量: {np.sum(ground_labels == 1)} ({np.sum(ground_labels == 1)/len(points)*100:.2f}%)")
    
    # 保存结果到文件（可选）
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        file_name = os.path.splitext(os.path.basename(input_path))[0]
        
        # 分离地面点和非地面点
        ground_points = ground_labels == 2
        non_ground_points = ground_labels == 1
        
        # 保存地面点
        ground_las = laspy.create(point_format=las.header.point_format, file_version=las.header.version)
        ground_las.points = las.points[ground_points]
        ground_las.write(os.path.join(output_dir, f"{file_name}_ground.las"))
        
        # 保存非地面点
        non_ground_las = laspy.create(point_format=las.header.point_format, file_version=las.header.version)
        non_ground_las.points = las.points[non_ground_points]
        non_ground_las.write(os.path.join(output_dir, f"{file_name}_non_ground.las"))
        print(f"结果已保存到: {output_dir}")
    
    # 使用Open3D可视化结果
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # 设置点云颜色
    colors = np.zeros((len(points), 3))
    
    # 地面点 - 绿色
    colors[ground_labels == 2] = [1, 0, 0]   # RGB - 红色
    
    # 非地面点 - 红色
    colors[ground_labels == 1] = [0.5, 0.5, 0.5]    # RGB - 灰色
    
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # 可视化参数设置
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='地面点云滤波结果', width=1200, height=900)
    vis.add_geometry(pcd)
    
    # 设置相机位置以更好查看
    ctr = vis.get_view_control()
    ctr.set_front([0, -1, 0.5])  # 设置相机位置
    ctr.set_lookat(pcd.get_center())  # 看向点云中心
    ctr.set_up([0, 0, 1])  # 设置上方向
    ctr.set_zoom(0.8)  # 缩放级别
    
    print("按 'Q' 或关闭窗口退出可视化...")
    
    # 添加坐标系
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=np.ptp(points, axis=0).max()/10, 
        origin=np.mean(points, axis=0))
    vis.add_geometry(mesh_frame)
    
    # 开始可视化
    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    # 配置输入文件
    input_las = r"G:\yzk\2_project\Point_cloud_filtering\5080_54435.las"  # 替换为你的LAS文件路径
    output_directory = r"G:\yzk\2_project\Point_cloud_filtering"
    
    # 处理并可视化LAS文件
    filter_and_visualize(input_las, output_directory)
