import numpy as np
import pyln


def main():
    pyln.utility.compile_numba()
    sphere = pyln.Sphere(texture=1)
    cube = pyln.StripedCube([-0.8, -0.8, -0.8], [0.8, 0.8, 0.8], 20)
    cylinder = pyln.Cylinder(0.4, -2.0, 2.0)
    shape = (
        (sphere * cube)
        - cylinder
        - cylinder.rotate_x(90)
        - cylinder.rotate_y(90)
    )
    images = []
    for index, i in enumerate(range(0, 90, 10)):
        scene = pyln.Scene()
        scene.add(shape.rotate_z(i))

        # define camera parameters
        eye = np.array([0, 6, 2], dtype=np.float64)  # camera position
        center = np.array([0, 0, 0], dtype=np.float64)  # camera looks at
        up = np.array([0, 0, 1], dtype=np.float64)  # up direction

        # define rendering parameters
        width = 750  # rendered width
        height = 750  # rendered height
        fovy = 20.0  # vertical field of view, degrees
        znear = 0.1  # near z plane
        zfar = 100.0  # far z plane
        step = 0.01  # how finely to chop the paths for visibility testing

        # compute 2D paths that depict the 3D scene
        paths = scene.render(
            eye, center, up, width, height, fovy, znear, zfar, step
        )
        images.append(paths.to_image(width, height))

    # save results
    images[0].save(
        "examples/images/csg.gif",
        save_all=True,
        append_images=images[1:],
        optimize=True,
        duration=100,
        loop=0,
    )


if __name__ == "__main__":
    main()
