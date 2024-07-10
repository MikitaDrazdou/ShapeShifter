import pyvista as pv
import os

class Preprocess:
    def __init__(self):
        pass

    def prepare(self, model_path, image_folder_path, start_num):
        filename = model_path
        mesh = pv.read(filename)
        screens = []

        rotation_point = [0, 0, 0]

        def save_screenshot(mesh, step, save_folder_path):
            plotter = pv.Plotter(off_screen=True, lighting="none")
            plotter.add_mesh(mesh, color="black")
            plotter.view_xy()
            plotter.disable_shadows()
            plotter.enable_parallel_projection()

            try:
                os.mkdir(save_folder_path)
            except Exception as e:
                pass

            screenshot = f"{save_folder_path}/model_combined_{step}.png"
            plotter.show(screenshot=screenshot)
            screens.append(screenshot)

        cnt = 0

        angles_x = [0, 90, 150, 240]
        angles_y = [0, 90, 150, 240]
        angles_z = [0, 120, 240]

        for i in range(len(angles_x)):
            for j in range(len(angles_y)):
                for k in range(len(angles_z)):
                    rotated_mesh = mesh.copy()
                    rotated_mesh.rotate_x(angles_x[i], point=rotation_point, inplace=True)
                    rotated_mesh.rotate_y(angles_y[j], point=rotation_point, inplace=True)
                    rotated_mesh.rotate_z(angles_z[k], point=rotation_point, inplace=True)
                    print(f"{cnt}: X: {angles_x[i]} Y: {angles_y[j]} Z: {angles_z[k]}")
                    cnt+=1
                    save_screenshot(rotated_mesh, cnt + start_num, image_folder_path)

        print("Screenshots saved:")
        for screenshot in screens:
            print(screenshot)