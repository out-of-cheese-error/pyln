import numpy as np
import pyln


def main():
    pyln.utility.compile_numba()
    # create a scene and add a bunch of cubes
    scene = pyln.Scene()
    for x in np.arange(-2, 2, 1):
        for y in np.arange(-2, 2, 1):
            z = np.random.random()
            vector = np.array([x, y, z], dtype=np.float64)
            scene.add(pyln.Cube(vector - 0.5, vector + 0.5))

    # define camera parameters
    eye = [6, 5, 3]  # camera position
    center = [0, 0, 0]  # camera looks at
    up = [0, 0, 1]  # up direction

    # define rendering parameters
    width = 1920  # rendered width
    height = 1200  # rendered height
    fovy = 30.0  # vertical field of view, degrees
    znear = 0.1  # near z plane
    zfar = 100.0  # far z plane
    step = 0.01  # how finely to chop the paths for visibility testing

    # compute 2D paths that depict the 3D scene
    paths = scene.render(
        eye, center, up, width, height, fovy, znear, zfar, step
    )

    # save results
    paths.write_to_svg(
        "examples/images/cubes.svg", width, height, background_color="white"
    )


if __name__ == "__main__":
    main()
