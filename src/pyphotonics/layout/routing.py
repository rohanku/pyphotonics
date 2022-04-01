import gdstk
import numpy as np
from tkinter import *
from pyphotonics.layout import utils, gui

tk = Tk()


class WaveguidePath:
    """
    GDS waveguide route.

    Parameters
    ----------
    points : list[numpy tuple]
        Coordinates of points defining the waveguide route.
    width : float
        Width of the waveguide in GDS units.
    r_vals : list[float]
        Possible bend radii in GDS units.

    Attributes
    ----------
    points : list[numpy tuple]
        Coordinates of points defining the waveguide route.
    width : float
        Width of the waveguide in GDS units.
    r_vals : list[float]
        Possible bend radii in GDS units.
    r_min : float
        Minimum bend radius in GDS units.
    N : int
        Number of points in path.
    bend_radii : list[float]
        Bend radius at each point, 0 at starting and ending vertices.

    Notes
    -----
    The path traced out by the points should make up the tangent line to the final path. Bends will be automatically computed upon export to GDS.
    """

    def __init__(self, points, width, r_vals):
        self.points = points
        self.width = width
        self.r_vals = r_vals
        self.r_min = min(r_vals)
        self.N = len(points)
        self.trim()
        self.bend_radii = [0] + [self.r_min] * (self.N - 2) + [0]
        self.ensure_r_min()
        self.maximize_bend_radii()

    def get_length(self, index):
        """
        Returns the length of the given path segment.

        Parameters
        ---------
        index : int
            Index of segment in path. Must be in the range [0, len(path)-1).

        Returns
        -------
        length : float
            Length of specified segment in path.
        """
        if index < 0 or index > self.N - 2:
            raise TypeError("Segment index must be between in [0, len(path)-1)")
        return utils.euclidean_distance(self.points[index], self.points[index + 1])

    def minimum_length(self, index):
        """
        Returns the minimum length of the given path segment such that the corresponding bend radii are valid.

        Parameters
        ---------
        index : int
            Index of segment in path. Must be in the range [0, len(path)-1).

        Returns
        -------
        min_length : float
            Minimum length of specificied segment in path.
        """
        if index < 0 or index > self.N - 2:
            raise TypeError("Segment index must be between in [0, len(path)-1)")
        path = self.points
        theta1 = (
            0
            if index == 0
            else utils.path_angle(path[index - 1], path[index], path[index + 1])
        )
        theta2 = (
            0
            if index == self.N - 2
            else utils.path_angle(path[index], path[index + 1], path[index + 2])
        )

        return (
            self.bend_radii[index] * np.abs(np.tan(theta1 / 2))
            + self.bend_radii[index + 1] * np.abs(np.tan(theta2 / 2))
            + 1e-5
        )

    def trim(self):
        """
        Deletes points that do not contribute to the waveguide geometry.
        """
        i = 0
        while i < self.N - 2:
            if np.isclose(
                0,
                utils.path_angle(
                    self.points[i], self.points[i + 1], self.points[i + 2]
                ),
            ):
                self.points.pop(i + 1)
                self.N -= 1
                continue
            i += 1

    def segment_drag(self, index, d):
        """
        Drag segment by the given perpendicular distance while maintaining its angle. Cannot drag the first or last segments.

        Parameters
        ----------
        index : int
            Index of segment in path. Must be in the range [1, len(path)-2).
        d : float
            Distance to drag segment.
        """
        if index < 1 or index > self.N - 3:
            raise TypeError(
                "Only segments with index in [1, len(path)-2) can be dragged"
            )
        path = self.points
        v_path = path[index + 1] - path[index]
        v_displacement = utils.perp(v_path)
        v_displacement /= np.linalg.norm(v_displacement)
        p_new = path[index] + v_displacement * d
        angle = utils.horizontal_angle(v_path)
        p1 = utils.line_intersect(
            p_new,
            angle,
            path[index],
            utils.horizontal_angle(path[index] - path[index - 1]),
        )
        p2 = utils.line_intersect(
            p_new,
            angle,
            path[index + 1],
            utils.horizontal_angle(path[index + 2] - path[index + 1]),
        )
        if p1 or p2 is None:
            raise TypeError(
                "No intersection found for shifted segment. This may be due to redundant points, call trim() first to fix this error"
            )
        path[index] = p1
        path[index + 1] = p2

    def ensure_r_min(self):
        """
        Shift path segments by minimum amount to ensure that there is space for each bend.
        """
        for i in range(1, self.N - 2):
            curr_len = self.get_length(i - 1)
            min_len = self.minimum_length(i - 1)
            theta = utils.path_angle(
                self.points[i - 1], self.points[i], self.points[i + 1]
            )
            if curr_len < min_len:
                self.segment_drag(i, (curr_len - min_len) / np.sin(theta))
        for i in range(self.N - 3, 0, -1):
            curr_len = self.get_length(i + 1)
            min_len = self.minimum_length(i + 1)
            theta = utils.path_angle(
                self.points[i + 2], self.points[i + 1], self.points[i]
            )
            if curr_len < min_len:
                self.segment_drag(i, (min_len - curr_len) / np.sin(theta))

    def maximize_bend_radii(self):
        """
        Assign the largest possible bend radius to each bend, prioritizing uniformity.
        """
        for i in range(1, len(self.r_vals)):
            for j in range(1, self.N - 1):
                curr_radius = self.bend_radii[j]
                self.bend_radii[j] = self.r_vals[i]
                if self.minimum_length(j) > self.get_length(j) or self.minimum_length(
                    j - 1
                ) > self.get_length(j - 1):
                    self.bend_radii[j] = curr_radius


def get_rough_paths(inputs, outputs, route_file=None, current_gds=None):
    """
    Returns a user specified route from an existing .route file or the GUI.

    Parameters
    ----------
    inputs : list[3-tuple]
        List of input ports represented by (x, y, angle), where angle is degrees counter-clockwise from the horizontal.
    outputs : list[3-tuple]
        List of output ports represented by (x, y, angle), where angle is degrees counter-clockwise from the horizontal.
    route_file : str
        Path to a .route file with the desired initial paths.
    current_gds : str
        Path to the GDS for which the routes are being generated.

    Returns
    -------
    paths : list[list[numpy tuple]]
        List of paths each consisting of a list of GDS coordinates.
    """
    pathing_gui = gui.PathingGUI(tk, inputs, outputs, current_gds)
    if not route_file:
        tk.mainloop()
    else:
        pathing_gui.set_route_file(open(route_file))
        pathing_gui.autoroute()
    if pathing_gui.autoroute_paths is None:
        print("Autoroute was not called, exiting...")
        return None
    return pathing_gui.autoroute_paths


def turn_port_route(
    port, r_min, target_angle, reverse=False, four_point_threshold=10.0
):
    """
    Returns a 3- or 4-point bend that turns the given port to the desired angle

    Parameters
    ----------
    port : 3-tuple
        Port represented by (x, y, angle), where angle is degrees counter-clockwise from the horizontal.
    r_min : double:
        The minimum radius for executing the turn.
    target_angle : double
        Angle in degrees counter-clockwise from the horizontal that the port should be turned to.
    reverse : bool
        Treat port as an output, returning the turn leading into the port from target_angle.
    four_point_threshold : double
        Bends with angles from [-180 + fpt, 180 - fpt] will have 3 points, outside of that range bends will have 4

    Returns
    -------
    bend : list[numpy tuple]
        3- or 4-point bend that turns the given port to the desired angle.
    """
    bend_angle = (target_angle - port[2]) % 360
    if bend_angle > 180:
        bend_angle -= 360
    bend_angle_rad = np.radians(bend_angle)
    port_angle_rad = np.radians(port[2])

    # Define the curve to arrive at the target angle using 4 points
    start = utils.get_port_coords(port)

    if np.isclose(bend_angle, 0):
        return [start]

    if (
        bend_angle < -180 + four_point_threshold
        or bend_angle > 180 - four_point_threshold
    ):
        l1 = np.abs(r_min * np.tan(bend_angle_rad / 4)) + 1e-5
        inter1 = np.array([l1 * np.cos(port_angle_rad), l1 * np.sin(port_angle_rad)])

        l2 = 2 * l1
        theta2 = port_angle_rad + bend_angle_rad / 2
        inter2 = inter1 + np.array([l2 * np.cos(theta2), l2 * np.sin(theta2)])

        theta3 = theta2 + bend_angle_rad / 2
        inter3 = inter2 + np.array([l1 * np.cos(theta3), l1 * np.sin(theta3)])

        if reverse:
            return [-inter3 + start, -inter2 + start, -inter1 + start, start]

        return [start, start + inter1, start + inter2, start + inter3]

    l = np.abs(r_min * np.tan(bend_angle_rad / 2)) + 1e-5
    inter1 = np.array([l * np.cos(port_angle_rad), l * np.sin(port_angle_rad)])
    target_angle_rad = np.radians(target_angle)
    inter2 = inter1 + np.array(
        [l * np.cos(target_angle_rad), l * np.sin(target_angle_rad)]
    )

    if reverse:
        return [-inter2 + start, -inter1 + start, start]

    return [start, start + inter1, start + inter2]


def direct_route(port1, port2, width, r_vals):
    """
    Returns a basic two segment route between two ports, adding segments as necessary to account for port angle.

    Parameters
    ----------
    port1 : 3-tuple
        Input port represented by (x, y, angle), where angle is degrees counter-clockwise from the horizontal
    port2 : 3-tuple
        Output port represented by (x, y, angle), where angle is degrees counter-clockwise from the horizontal
    width : float
        Width of waveguides in GDS units
    r_vals : list[float]
        Possible bend radii in GDS units

    Returns
    -------
    path : WaveguidePath
        WaveguidePath between input and output port.
    """
    if len(r_vals) == 0 or min(r_vals) <= 0:
        raise ValueError("Minimum radius must be positive")

    # Determine the best set of directions for the given ports to be bent to minimize total bend angle
    dirs = utils.get_perpendicular_directions(port1, port2)
    best_dir = dirs[
        np.argmin(
            list(
                map(
                    lambda x: abs((port1[2] - x[0]) % 360)
                    + abs((port2[2] - x[1]) % 360),
                    dirs,
                )
            )
        )
    ]

    # Turn ports to the desired angle
    points1 = turn_port_route(port1, min(r_vals), best_dir[0])
    points2 = turn_port_route(
        port2, min(r_vals), best_dir[1], reverse=True
    )  # Reverse for output

    # Add intermediate point for manhattan routing
    coords1 = points1[-1] if best_dir[0] == 90 or best_dir[0] == -90 else points2[0]
    coords2 = points2[0] if best_dir[0] == 90 or best_dir[0] == -90 else points1[-1]

    return WaveguidePath(
        points1 + [np.array([coords1[0], coords2[1]])] + points2, width, r_vals
    )


def user_route(
    inputs, outputs, width, r_vals, d_min, route_file=None, current_gds=None
):
    """
    Returns a user specified route that has been modified to fit the given specifications. Opens a routing GUI if necessary to allow for user input.

    Parameters
    ----------
    inputs : list[3-tuple]
        List of input ports represented by (x, y, angle), where angle is degrees counter-clockwise from the horizontal.
    outputs : list[3-tuple]
        List of output ports represented by (x, y, angle), where angle is degrees counter-clockwise from the horizontal.
    width : double
        Width of waveguides in GDS units.
    r_vals : list of doubles
        Possible bend radii in GDS units.
    d_min : double
        Minimum distance between waveguides in GDS units.
    route_file : str
        Path to a .route file with the desired initial paths.
    current_gds : str
        Path to the GDS for which the routes are being generated.

    Returns
    -------
    path : WaveguidePath
        WaveguidePath between input and output port
    """

    if len(inputs) != len(outputs):
        raise ValueError("Input and output port arrays must have the same length")

    if len(r_vals) == 0 or min(r_vals) <= 0:
        raise ValueError("Minimum radius must be positive")

    initial_paths = get_rough_paths(inputs, outputs, route_file, current_gds)
    if initial_paths is None:
        raise TypeError("No initial paths specified")

    final_paths = []
    for i in range(len(inputs)):
        path = initial_paths[i]
        curr_dir = utils.manhattan_angle(utils.horizontal_angle(path[1] - path[0]))
        manhattan_path = turn_port_route(inputs[i], min(r_vals), np.degrees(curr_dir))
        for j in range(1, len(path) - 2):
            manhattan_path.append(
                utils.ray_project(manhattan_path[-1], curr_dir, path[j])
            )
            curr_dir = utils.manhattan_angle(
                utils.horizontal_angle(path[j + 1] - path[j])
            )
        final_dir = utils.manhattan_angle(utils.horizontal_angle(path[-1] - path[-2]))
        output_turn = turn_port_route(
            outputs[i], min(r_vals), np.degrees(final_dir), reverse=True
        )
        intermediate = utils.ray_intersect(
            manhattan_path[-1],
            curr_dir,
            output_turn[0],
            utils.reverse_angle(final_dir),
        )
        if intermediate is None:
            raise TypeError("Provided path is not a valid Manhattan route")
        manhattan_path.extend([intermediate, *output_turn])
        final_paths.append(WaveguidePath(manhattan_path, width, r_vals))
    return final_paths


def write_paths_to_gds(paths, output_file, layer=0, datatype=0, geometry="bend"):
    """
    Generates a set of paths with waveguides connecting inputs ports to output ports.

    Parameters
    ----------
    paths : list[WaveguidePath]
        List of WaveguidePaths to be written.
    output_file : str
        Path to output GDS file.
    layer : int
        Desired GDS layer for waveguides.
    datatype : int
        Datatype label for GDS layer.
    geometry : 'bend' or 'rectilinear'
        "bend" adds bends of specified radius at each vertex during export, "rectilinear" writes the path as is without bends.

    Returns
    -------
    paths : list[WaveguidePath]
        List of WaveguidePaths between their corresponding inputs and outputs.
    """
    if geometry == "bend":
        for path in paths:
            points = path.points
            N = len(points)
            prev_tangent_len = 0
            for i in range(N - 1):
                segment_len = path.get_length(i)
                if i == N - 2:
                    nd.strt(
                        length=segment_len - prev_tangent_len,
                        width=path.width,
                        layer=(layer, datatype),
                    ).put()
                else:
                    angle = utils.path_angle(points[i], points[i + 1], points[i + 2])
                    interior_angle = np.pi - np.abs(angle)
                    new_tangent_len = (
                        0
                        if np.isclose(interior_angle, 0.0)
                        else path.bend_radii[i + 1] / np.tan(interior_angle / 2)
                    )
                    if i == 0:
                        starting_angle = utils.path_angle(
                            (points[i][0] - 1, points[i][1]), points[i], points[i + 1]
                        )
                        nd.strt(
                            length=segment_len - prev_tangent_len - new_tangent_len,
                            width=path.width,
                            layer=(layer, datatype),
                        ).put(
                            float(points[i][0]),
                            float(points[i][1]),
                            np.degrees(starting_angle),
                        )
                    else:
                        nd.strt(
                            length=segment_len - prev_tangent_len - new_tangent_len,
                            width=path.width,
                            layer=(layer, datatype),
                        ).put()
                    nd.bend(
                        angle=np.degrees(angle),
                        radius=path.bend_radii[i + 1],
                        width=path.width,
                        layer=(layer, datatype),
                    ).put()
                    prev_tangent_len = new_tangent_len
        nd.export_gds(filename=output_file)
    elif geometry == "rectilinear":
        lib = gdstk.Library()
        cell = lib.new_cell("AUTOROUTE")
        for path in paths:
            fp = gdstk.FlexPath(path.points, path.width, layer=layer, datatype=datatype)
            cell.add(fp)
        lib.write_gds(output_file)
    else:
        raise TypeError(f"Invalid geometry type: {geometry}")

