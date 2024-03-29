import pyln


def main():
    # compile
    pyln.utility.compile_numba()

    scene = pyln.Scene()
    box = pyln.Box([-2, -2, -4], [2, 2, 2])
    scene.add(
        pyln.Function(lambda x, y: -1 / (x * x + y * y), box, pyln.Direction.Below)
    )

    # define camera parameters
    eye = [3.0, 0.0, 3.0]  # camera position
    center = [1.1, 0.0, 0.0]  # camera looks at
    up = [0.0, 0.0, 1.0]  # up direction

    # define rendering parameters
    width = 1024  # rendered width
    height = 1024  # rendered height
    fovy = 50.0  # vertical field of view, degrees
    znear = 0.1  # near z plane
    zfar = 100.0  # far z plane
    step = 0.01  # how finely to chop the paths for visibility testing

    # compute 2D paths that depict the 3D scene
    paths = scene.render(eye, center, up, width, height, fovy, znear, zfar, step)
    # save results
    paths.write_to_svg(
        "examples/images/function.svg", width, height, background_color="white"
    )


if __name__ == "__main__":
    main()
