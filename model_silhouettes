import pyvista as pv

filename = "IronMan.obj"
mesh = pv.read(filename)
screens = []

rotation_point = [0, 0, 0]

steps = 3
angle_increment = 20

def save_screenshot(mesh, step):
    plotter = pv.Plotter(off_screen=True, lighting="none")
    plotter.add_mesh(mesh, color="black")
    plotter.view_xy()
    plotter.disable_shadows()
    plotter.enable_parallel_projection()
    screenshot = f"model_combined_{step}.png"
    plotter.show(screenshot=screenshot)
    screens.append(screenshot)

cnt = 0
for i in range(steps):
    rotated_mesh = mesh.copy()
    rotated_mesh.rotate_x(angle_increment*i, point=rotation_point, inplace=True)
    for j in range(steps):
        rotated_mesh.rotate_y(angle_increment*j, point=rotation_point, inplace=True)
        for k in range(steps):
            rotated_mesh.rotate_z(angle_increment*k, point=rotation_point, inplace=True)
            cnt+=1
            save_screenshot(rotated_mesh, cnt)

print("Screenshots saved:")
for screenshot in screens:
    print(screenshot)
