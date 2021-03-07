import numpy as np
import pyln


def main():
    pyln.utility.compile_numba()
    # create a scene
    scene = pyln.Scene()
    mesh = pyln.Mesh.from_stl("examples/suzanne.stl")

    mesh.fit_inside(
        pyln.Box([-1, -1, -1], [1, 1, 1]), np.array([0.5, 0.5, 0.5])
    )
    cubes = mesh.voxelize(1.0 / 64)
    for cube in cubes:
        scene.add(cube)

    # define camera parameters
    eye = np.array([-1, -2, 0], dtype=np.float64)  # camera position
    center = np.array([0, 0, 0], dtype=np.float64)  # camera looks at
    up = np.array([0, 0, 1], dtype=np.float64)  # up direction

    # define rendering parameters
    width = 1024 / 2  # rendered width
    height = 1024 / 2  # rendered height
    fovy = 60.0  # vertical field of view, degrees
    znear = 0.1  # near z plane
    zfar = 100.0  # far z plane
    step = 0.01  # how finely to chop the paths for visibility testing

    # compute 2D paths that depict the 3D scene
    paths = scene.render(
        eye, center, up, width, height, fovy, znear, zfar, step
    )

    # save results
    paths.write_to_svg("examples/images/voxelize.svg", width, height)


if __name__ == "__main__":
    main()
