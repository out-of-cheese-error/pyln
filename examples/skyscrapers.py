import numpy as np

import pyln


def main():
    pyln.utility.compile_numba()
    scene = pyln.Scene()
    n = 15
    for x in np.arange(-n, n):
        for y in np.arange(-n, n):
            if x == 2 and y == 1:
                continue
            p = np.random.random() * 0.25 + 0.2
            fx = x + (np.random.random() * 0.5 - 0.25)
            fy = y + (np.random.random() * 0.5 - 0.25)
            fz = np.random.random() * 3 + 1
            scene.add(pyln.Cube([fx - p, fy - p, 0], [fx + p, fy + p, fz]))
    # define camera parameters
    eye = [1.75, 1.25, 6.0]  # camera position
    center = [0.0, 0.0, 0.0]  # camera looks at
    up = [0.0, 0.0, 1.0]  # up direction

    # define rendering parameters
    width = 1024  # rendered width
    height = 1024  # rendered height
    fovy = 100.0  # vertical field of view, degrees
    znear = 0.1  # near z plane
    zfar = 100.0  # far z plane
    step = 0.005  # how finely to chop the paths for visibility testing

    # compute 2D paths that depict the 3D scene
    paths = scene.render(eye, center, up, width, height, fovy, znear, zfar, step)

    # save results
    paths.write_to_svg(
        "examples/images/skyscrapers.svg",
        width,
        height,
        background_color="white",
    )


if __name__ == "__main__":
    main()
