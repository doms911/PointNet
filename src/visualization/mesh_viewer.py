import open3d as o3d

PATH = "data/raw/ModelNet10/toilet/test/toilet_0403.off"

mesh = o3d.io.read_triangle_mesh(PATH)
mesh.compute_vertex_normals()

# pcd = mesh.sample_points_uniformly(number_of_points=2048)

# o3d.visualization.draw_geometries([pcd])
print(mesh)
print("Vertices:", len(mesh.vertices))
print("Triangles:", len(mesh.triangles))

o3d.visualization.draw_geometries([mesh])