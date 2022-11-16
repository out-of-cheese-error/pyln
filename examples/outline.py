import numpy as np

import pyln


def main():
    pyln.utility.compile_numba()
    # define camera parameters
    eye = [8.0, 8.0, 8.0]  # camera position
    center = [0.0, 0.0, 0.0]  # camera looks at
    up = [0.0, 0.0, 1.0]  # up direction
    scene = pyln.Scene()
    n = 10
    for x in np.arange(-n, n):
        for y in np.arange(-n, n):
            scene.add(pyln.OutlineSphere(eye, up, [x, y, np.random.random()], 0.45))

    # define rendering parameters
    width = 1920  # rendered width
    height = 1200  # rendered height
    fovy = 50.0  # vertical field of view, degrees
    znear = 0.1  # near z plane
    zfar = 100.0  # far z plane
    step = 0.01  # how finely to chop the paths for visibility testing

    # compute 2D paths that depict the 3D scene
    paths = scene.render(eye, center, up, width, height, fovy, znear, zfar, step)

    # save results
    paths.write_to_svg(
        "examples/images/outline.svg", width, height, background_color="white"
    )


if __name__ == "__main__":
    main()
