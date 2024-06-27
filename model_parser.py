import pyvista as pv

filename = "something2.obj"
mesh = pv.read(filename)


for i in range(6):
    plotter = pv.Plotter(off_screen=True, lighting="none")

    axes = pv.Axes()

    rot = mesh.rotate_y(15 * i, point=axes.origin, inplace=True)
    plotter.add_mesh(rot, color="black")

    plotter.view_xy()
    plotter.disable_shadows()
    plotter.enable_parallel_projection()
    
    plotter.show(screenshot=f"model{i}.png")