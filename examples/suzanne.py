import pyln


def main():
    pyln.utility.compile_numba()
    # create a scene
    scene = pyln.Scene()
    mesh = pyln.Mesh.from_obj("examples/suzanne.obj")
    mesh.unit_cube()
    scene.add(mesh.rotate_y(30))

    # define camera parameters
    eye = [-0.5, 0.5, 2.0]  # camera position
    center = [0.0, 0.0, 0.0]  # camera looks at
    up = [0.0, 1.0, 0.0]  # up direction

    # define rendering parameters
    width = 1024  # rendered width
    height = 1024  # rendered height
    fovy = 35.0  # vertical field of view, degrees
    znear = 0.1  # near z plane
    zfar = 100.0  # far z plane
    step = 0.01  # how finely to chop the paths for visibility testing

    # compute 2D paths that depict the 3D scene
    paths = scene.render(eye, center, up, width, height, fovy, znear, zfar, step)

    # save results
    paths.write_to_svg(
        "examples/images/suzanne.svg", width, height, background_color="white"
    )


if __name__ == "__main__":
    main()
