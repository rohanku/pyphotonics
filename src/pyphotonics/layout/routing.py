import gdstk
import numpy as np
from tkinter import *
from pyphotonics.layout import utils, gui
from scipy.integrate import solve_ivp


class WaveguideGeometry:
    """
    Waveguide shape.

    Attributes
    ----------
    width : float
        Base width of waveguide in GDS units.
    geometry : 'slab' or 'ridge'
        Geometry of waveguide. Ridge waveguides are currently unimplemented.
    ridge_width : float
        Width of partial etch mask for ridge in GDS units.
    """

    def __init__(self, width, kind="slab", ridge_width=None):
        if ridge_width is None:
            if kind == "ridge":
                raise ValueError("No ridge width specified with ridge geometry")
        else:
            if ridge_width > width:
                raise ValueError(
                        "Ridge width cannot be larger than the width of the waveguide"
                        )
            if kind != "ridge":
                print("Warning: ridge width specified with non-ridge geometry")

        self.width = width
        self.kind = kind
        self.ridge_width = ridge_width

    def __eq__(self, other):
        if self.kind == "slab":
            return self.kind == other.kind and np.isclose(self.width, other.width)
        else:
            return self.kind == other.kind and np.allclose(
                    [self.width, self.ridge_width], [other.width, other.ridge_width]
                    )


class Port:
    """
    Connection port for waveguide routing.

    Attributes
    ----------
    x : float
        GDS x coordinate.
    y : float
        GDS y coordinate.
    angle : float
        Angle from positive x axis in radians.
    geometry : WaveguideGeometry
        Geometry of waveguide at port.
    """

    def __init__(self, x, y, angle, geometry=None):
        self.x = x
        self.y = y
        self.angle = angle
        self.geometry = geometry

    def __eq__(self, other):
        return np.allclose(
                np.array([self.x, self.y]), np.array([other.x, other.y])
                ) and utils.angle_close(self.angle, other.angle)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((round(self.x, 3), round(self.y, 3), round(self.angle, 3)))

    def __str__(self):
        return f"{self.x} {self.y} {self.angle} {self.geometry}"


class WaveguidePath:
    """
    GDS waveguide route.

    Parameters
    ----------
    router : Router
        Waveguide router class.
    points : list[numpy tuple]
        Coordinates of points defining the waveguide route.
    wg_geometry : WaveguideGeometry
        Geometry of the waveguide.
    r_vals : list[float]
        Possible bend radii in GDS units.
    input_geometry: WaveguideGeometry
        Geometry of input port.
    output_geometry: WaveguideGeometry
        Geometry of output port.

    Attributes
    ----------
    N : int
        Number of points in path.
    points : list[numpy tuple]
        Coordinates of points defining the waveguide route.
    wg_geometry : WaveguideGeometry
        Geometry of the waveguide.
    r_vals : list[float]
        Possible bend radii in GDS units.
    r_min : float
        Minimum bend radius in GDS units.
    bend_radii : list[float]
        Bend radius at each point, 0 at starting and ending vertices.
    input_geometry: WaveguideGeometry
        Geometry of input port.
    output_geometry: WaveguideGeometry
        Geometry of output port.
    input_taper_len: float
        Length of input taper
    output_taper_len: float
        Length of output taper

    Notes
    -----
    The path traced out by the points should make up the tangent line to the final path. Bends will be automatically computed upon export to GDS.
    """

    def __init__(
            self,
            router,
            points,
            wg_geometry,
            r_vals,
            input_geometry=None,
            output_geometry=None,
            ):
        self.router = router
        self.points = points
        self.wg_geometry = wg_geometry
        self.r_vals = r_vals
        self.input_geometry = wg_geometry if input_geometry is None else input_geometry
        self.output_geometry = (
                wg_geometry if output_geometry is None else output_geometry
                )
        self.r_min = min(r_vals)
        self.N = len(points)
        self.trim()
        self.bend_radii = [0] + [self.r_min] * (self.N - 2) + [0]
        (
                self.input_taper_poly1,
                self.input_taper_poly2,
                self.input_taper_end,
                ) = router.calculate_taper(
                        input_geometry,
                        wg_geometry,
                        points[0],
                        utils.horizontal_angle(points[1] - points[0]),
                        )
        self.input_taper_len = np.linalg.norm(points[0] - self.input_taper_end)
        (
                self.output_taper_poly1,
                self.output_taper_poly2,
                self.output_taper_end,
                ) = router.calculate_taper(
                        output_geometry,
                        wg_geometry,
                        points[-1],
                        utils.horizontal_angle(points[-2] - points[-1]),
                        )
        self.output_taper_len = np.linalg.norm(points[-1] - self.output_taper_end)
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
            raise ValueError("Segment index must be between in [0, len(path)-1)")
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
            raise ValueError("Segment index must be between in [0, len(path)-1)")
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
            (self.input_taper_len if index == 0 else 0)
            + self.bend_radii[index] * np.abs(np.tan(theta1 / 2))
            + self.bend_radii[index + 1] * np.abs(np.tan(theta2 / 2))
            + (self.output_taper_len if index == self.N - 2 else 0)
            + 1e-6
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
            raise ValueError(
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
        if p1 is None or p2 is None:
            raise ValueError(
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


class Router:
    """
    Waveguide router class.

    Attributes
    ----------
    wavelength : float
        Operation wavelength of device.
    n_core : float
        Refractive index of waveguide core.
    n_cladding : float
        Refractive index of waveguide cladding.
    alpha : float
        Proportionality constant for width of adiabatic tapers.
    """
    def __init__(self, wavelength=1.762, n_core=3.67, n_cladding=1.44, alpha=0.9):
        self.wavelength = wavelength
        self.n_core = n_core
        self.n_cladding = n_cladding
        self.alpha = alpha

    def calculate_taper(self, input_geometry, output_geometry, start, angle):
        """
        Computes a taper with the given parameters for export to GDS.

        Parameters
        ----------
        input_geometry : WaveguideGeometry
            Input geometry of taper.
        output_geometry: WaveguideGeometry
            Output geometry of taper.
        start: numpy tuple
            Coordinates of taper start.
        angle: float
            Angle of taper in radians counter-clockwise from the horizontal.

        Returns
        -------
        poly1 : list[numpy tuple]
            GDS polygon for first layer of taper.
        poly2 : list[numpy tuple]
            GDS polygon for second layer of taper.
        end : numpy tuple
            Coordinate of midpoint of end of taper.
        """
        if np.isclose(input_geometry.width, output_geometry.width):
            # Return no taper if widths are already aligned
            return None, None, start

        # General differential equation for adiabatic taper: https://opg.optica.org/prj/fulltext.cfm?uri=prj-2-3-A41&id=284545
        def adiabatic_taper_func(t, y):
            return np.tan(np.clip(self.alpha * self.wavelength / 4 / y / self.n_core, 0, np.pi/2 * 0.8))

        # Differential equation for tapering up waveguide width
        def outward_taper_func(t, y):
            return adiabatic_taper_func(t, y)

        # Differential equation for tapering down waveguide width
        def inward_taper_func(t, y):
            return -adiabatic_taper_func(t, y)

        # Terminate differential equation solver once waveguide reaches desired width
        def termination_event(t, y):
            return y[0] - output_geometry.width / 2
        termination_event.terminal = True

        taper_func = inward_taper_func if input_geometry.width > output_geometry.width else outward_taper_func
        # Choose conservative time span using minimum derivative
        t_span = (
            0,
            abs((input_geometry.width - output_geometry.width)
            / taper_func(0, max(input_geometry.width, output_geometry.width) / 2)
            / 2),
        )
        # Solve initial value problem to generate taper geometry
        sol = solve_ivp(
            taper_func,
            t_span,
            [input_geometry.width / 2],
            events=termination_event,
            max_step=t_span[1]/100,
        )
        sol_y = sol.y.flatten()
        sol_y[-1] = output_geometry.width / 2

        # Generate taper ridge
        dir_vec = np.array([np.cos(angle), np.sin(angle)])
        perp_vec = utils.perp(dir_vec)
        poly2 = None
        if (
            input_geometry.kind == "slab"
            and output_geometry.kind == "ridge"
            and input_geometry.width == output_geometry.ridge_width
        ):
            poly2 = [
                start + input_geometry.width / 2 * perp_vec,
                start - input_geometry.width / 2 * perp_vec,
                start - input_geometry.width / 2 * perp_vec + sol.t[-1] * dir_vec,
                start + input_geometry.width / 2 * perp_vec + sol.t[-1] * dir_vec,
            ]
        elif (
            input_geometry.kind == "ridge"
            and output_geometry.kind == "slab"
            and input_geometry.ridge_width == output_geometry.width
        ):
            poly2 = [
                start + output_geometry.width / 2 * perp_vec,
                start - output_geometry.width / 2 * perp_vec,
                start - output_geometry.width / 2 * perp_vec + sol.t[-1] * dir_vec,
                start + output_geometry.width / 2 * perp_vec + sol.t[-1] * dir_vec,
            ]
        elif input_geometry.kind == output_geometry.kind:
            if input_geometry.kind == "ridge" and input_geometry.ridge_width == output_geometry.ridge_width:
                poly2 = [
                    start + input_geometry.ridge_width / 2 * perp_vec,
                    start - input_geometry.ridge_width / 2 * perp_vec,
                    start - input_geometry.ridge_width / 2 * perp_vec + sol.t[-1] * dir_vec,
                    start + input_geometry.ridge_width / 2 * perp_vec + sol.t[-1] * dir_vec,
                ]
        else:
            # Return no taper for other geometries
            return None, None, start

        # Generate rotation matrix to rotate taper to desired angle
        rot_matrix = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )
        # Convert solution to list of (x, y) coordinates and add in other side of taper
        poly1 = list(map(lambda x: np.array(x), zip(sol.t, sol_y)))
        poly1 = list(
            map(
                lambda x: rot_matrix @ x + start,
                poly1 + list(map(lambda x: (x[0], -x[1]), poly1))[::-1],
            )
        )
        return poly1, poly2, start + sol.t[-1] * dir_vec

    def get_potential_ports(self, gds, geometries, bbox=None):
        """
        Finds potential ports in the provided GDS file using the geometry of the connecting waveguides.

        Parameters
        ----------
        gds : str
            Path to the GDS for which the routes are being generated.
        geometries: list[WaveguideGeometry]
            Geometry of waveguide connecting ports.
        bbox: 4-tuple
            (x0, y0, x1, y1) tuple where (x0, y0) and (x1, y1) are the bottom-left and top-right corners, respectively, of the area in which to look for ports.

        Returns
        -------
        ports : list[Port]
            List of ports that correspond to the given geometry.
        """
        lib = gdstk.read_gds(gds)
        potential_ports = set()
        for geometry in geometries:
            slab_geometry = WaveguideGeometry(geometry.ridge_width)
            if geometry.kind == "ridge":
                for cell in lib.cells:
                    for poly in cell.get_polygons():
                        # Consider only polygons that overlap with the desired bounding box
                        if bbox is not None and not utils.bbox_overlap(
                            bbox, tuple([t for tup in poly.bounding_box() for t in tup])
                        ):
                            continue

                        N = len(poly.points)
                        for i in range(N):
                            # Consider only potential ports that lie within the desired bounding box
                            midpoint = (poly.points[i] + poly.points[(i + 1) % N]) / 2
                            if bbox is not None and not utils.in_bbox(
                                bbox, midpoint[0], midpoint[1]
                            ):
                                continue

                            # Find ports that match the ridge width of the desired geometry
                            edge_len = utils.euclidean_distance(
                                poly.points[i], poly.points[(i + 1) % N]
                            )
                            if np.isclose(edge_len, geometry.ridge_width):
                                v = utils.perp(poly.points[(i + 1) % N] - poly.points[i])
                                first_ccw = (
                                    utils.path_angle(
                                        poly.points[i - 1],
                                        poly.points[i],
                                        poly.points[(i + 1) % N],
                                    )
                                    >= 0
                                )
                                second_ccw = (
                                    utils.path_angle(
                                        poly.points[i],
                                        poly.points[(i + 1) % N],
                                        poly.points[(i + 2) % N],
                                    )
                                    >= 0
                                )

                                # Determine direction of perpendicular vector
                                if first_ccw != second_ccw:
                                    continue
                                if first_ccw and second_ccw:
                                    v = -v

                                port = Port(
                                    midpoint[0],
                                    midpoint[1],
                                    utils.horizontal_angle(v),
                                    geometry=slab_geometry,
                                )
                                potential_ports.add(port)

            for cell in lib.cells:
                for poly in cell.get_polygons():
                    # Consider only polygons that overlap with the desired bounding box
                    if bbox is not None and not utils.bbox_overlap(
                        bbox, tuple(t for tup in poly.bounding_box() for t in tup)
                    ):
                        continue

                    N = len(poly.points)
                    for i in range(N):
                        # Consider only potential ports that lie within the desired bounding box
                        midpoint = (poly.points[i] + poly.points[(i + 1) % N]) / 2
                        if bbox is not None and not utils.in_bbox(
                            bbox, midpoint[0], midpoint[1]
                        ):
                            continue

                        # Find ports that match the full width of the desired geometry
                        edge_len = utils.euclidean_distance(
                            poly.points[i], poly.points[(i + 1) % N]
                        )
                        if np.isclose(edge_len, geometry.width, atol=1e-2):
                            v = utils.perp(poly.points[(i + 1) % N] - poly.points[i])
                            first_ccw = (
                                utils.path_angle(
                                    poly.points[i - 1],
                                    poly.points[i],
                                    poly.points[(i + 1) % N],
                                )
                                >= 0
                            )
                            second_ccw = (
                                utils.path_angle(
                                    poly.points[i],
                                    poly.points[(i + 1) % N],
                                    poly.points[(i + 2) % N],
                                )
                                >= 0
                            )

                            # Determine direction of perpendicular vector
                            if first_ccw != second_ccw:
                                continue
                            if first_ccw and second_ccw:
                                v = -v

                            port = Port(
                                midpoint[0],
                                midpoint[1],
                                utils.horizontal_angle(v),
                                geometry=geometry,
                            )

                            if geometry.kind == 'slab':
                                potential_ports.add(port)
                            elif port in potential_ports:
                                # Reinsert port with ridge geometry instead of slab
                                potential_ports.remove(port)
                                potential_ports.add(port)

        final_ports = set()
        for port in potential_ports:
            # Remove ports that are already connected to one another
            flipped_port = utils.reverse_port(port)
            if flipped_port not in potential_ports:
                final_ports.add(port)

        return final_ports

    def get_rough_paths(
        self,
        inputs=[],
        outputs=[],
        bbox=None,
        geometries=None,
        route_file=None,
        current_gds=None,
    ):
        """
        Returns a user specified route from an existing .route file or the GUI.

        Parameters
        ----------
        inputs : list[Port]
            List of input ports.
        outputs : list[Port]
            List of output ports.
        bbox : 4-tuple
            (x0, y0, x1, y1) tuple where (x0, y0) and (x1, y1) are the bottom-left and top-right corners, respectively, of the area in which the paths will be found. Automatically calculated from provided inputs and outputs unless explicitly specified.
        geometries : list[WaveguideGeometry]
            If provided, allows the GUI to detect unspecified potential ports.
        route_file : str
            Path to a .route file with the desired initial paths.
        current_gds : str
            Path to the GDS for which the routes are being generated.

        Returns
        -------
        inputs : list[Port]
            List of input ports corresponding to returned paths.
        outputs : list[Port]
            List of output ports corresponding to returned paths.
        paths : list[list[numpy tuple]]
            List of paths each consisting of a list of GDS coordinates.
        """

        tk = Tk()
        if bbox is None:
            if len(inputs) == 0:
                raise ValueError(
                    "If bounding box is not specified, must provide at least one input/output pair"
                )
            bbox = utils.get_bounding_box(
                list(map(utils.get_port_coords, inputs))
                + list(map(utils.get_port_coords, outputs)),
                padding=500,
            )
        potential_ports = []
        if not route_file and geometries is not None:
            potential_ports = (
                []
                if current_gds is None
                else list(
                    filter(
                        lambda x: x not in inputs
                        and utils.reverse_port(x) not in outputs,
                        self.get_potential_ports(current_gds, geometries, bbox=bbox),
                    )
                )
            )
        pathing_gui = gui.PathingGUI(
            tk, inputs, outputs, bbox, potential_ports, current_gds=current_gds
        )

        if not route_file:
            tk.mainloop()
        else:
            pathing_gui.set_route_file(open(route_file))
            pathing_gui.autoroute()
        if pathing_gui.autoroute_paths is None:
            print("Autoroute was not called, exiting...")
            return pathing_gui.inputs, pathing_gui.outputs, None
        return pathing_gui.inputs, pathing_gui.outputs, pathing_gui.autoroute_paths

    def turn_port_route(
        self,
        port,
        r_min,
        target_angle,
        reverse=False,
        four_point_threshold=np.pi / 18.0,
    ):
        """
        Returns a 3- or 4-point bend that turns the given port to the desired angle.

        Parameters
        ----------
        port : Port
            Port to turn to target angle.
        r_min : float
            The minimum radius for executing the turn.
        target_angle : float
            Angle in radians counter-clockwise from the horizontal that the port should be turned to.
        reverse : bool
            Treat port as an output, returning the turn leading into the port from target_angle.
        four_point_threshold : float
            Bends with angles from [-pi + fpt, pi - fpt] will have 3 points, outside of that range bends will have 4

        Returns
        -------
        bend : list[numpy tuple]
            3- or 4-point bend that turns the given port to the desired angle.
        """
        bend_angle = utils.normalize_angle(target_angle - port.angle)

        # Define the curve to arrive at the target angle using 4 points
        start = utils.get_port_coords(port)

        if np.isclose(bend_angle, 0):
            return [start]

        if (
            bend_angle < -np.pi + four_point_threshold
            or bend_angle > np.pi - four_point_threshold
        ):
            l1 = np.abs(r_min * np.tan(bend_angle / 4)) + 1e-6
            inter1 = np.array([l1 * np.cos(port.angle), l1 * np.sin(port.angle)])

            l2 = 2 * l1
            theta2 = port.angle + bend_angle / 2
            inter2 = inter1 + np.array([l2 * np.cos(theta2), l2 * np.sin(theta2)])

            theta3 = theta2 + bend_angle / 2
            inter3 = inter2 + np.array([l1 * np.cos(theta3), l1 * np.sin(theta3)])

            if reverse:
                return [-inter3 + start, -inter2 + start, -inter1 + start, start]

            return [start, start + inter1, start + inter2, start + inter3]

        l = np.abs(r_min * np.tan(bend_angle / 2)) + 1e-6
        inter1 = np.array([l * np.cos(port.angle), l * np.sin(port.angle)])
        inter2 = inter1 + np.array([l * np.cos(target_angle), l * np.sin(target_angle)])

        if reverse:
            return [-inter2 + start, -inter1 + start, start]

        return [start, start + inter1, start + inter2]

    def direct_route(self, port1, port2, geometry, r_vals, x_first=None):
        """
        Returns a basic two segment Manhattan route between two ports, adding segments as necessary to account for port angle.

        Parameters
        ----------
        port1 : Port
            Input port.
        port2 : Port
            Output port.
        geometry : WaveguideGeometry
            Geometry of waveguide.
        r_vals : list[float]
            Possible bend radii in GDS units
        x_first : bool or None
            Determines whether the Manhattan bend traverses the x axis or y axis first. If None, the direction that minimizes bends is chosen.

        Returns
        -------
        path : WaveguidePath
            WaveguidePath between input and output port.

        Warning
        -------
        :meth:`direct_route` does not account for waveguide crossings or obstacles and simply connects two ports in the most efficient way possible. Crossings can be avoided by offsetting ports slightly from one another or using the ``x_first`` parameter. It may also not work for cases where the ports cannot be connected by a simple two segment path.
        """
        if len(r_vals) == 0 or min(r_vals) <= 0:
            raise ValueError("Minimum radius must be positive")

        port_coords1 = utils.get_port_coords(port1)
        port_coords2 = utils.get_port_coords(port2)
        if np.isclose(port_coords1[0], port_coords2[0]) or np.isclose(
            port_coords1[1], port_coords2[1]
        ):
            horiz_angle = utils.horizontal_angle(port_coords2 - port_coords1)
            if np.isclose(horiz_angle, port1.angle) and np.isclose(
                horiz_angle, port2.angle
            ):
                return WaveguidePath(
                    self,
                    [port_coords1, port_coords2],
                    geometry,
                    r_vals,
                    input_geometry=port1.geometry,
                    output_geometry=port2.geometry,
                )

        # Determine the best set of directions for the given ports to be bent to minimize total bend angle
        dirs = utils.get_perpendicular_directions(port1, port2)
        best_dir = (
            dirs[
                np.argmin(
                    list(
                        map(
                            lambda x: abs(utils.normalize_angle(port1.angle - x[0]))
                            + abs(utils.normalize_angle(port2.angle - x[1])),
                            dirs,
                        )
                    )
                )
            ]
            if x_first is None
            else dirs[0 if x_first else 1]
        )

        # Turn ports to the desired angle
        points1 = self.turn_port_route(port1, min(r_vals), best_dir[0])
        points2 = self.turn_port_route(
            port2, min(r_vals), best_dir[1], reverse=True
        )  # Reverse for output

        # Add intermediate point for manhattan routing
        coords1, coords2 = (
            (points1[-1], points2[0])
            if np.isclose(best_dir[0], np.pi / 2.0)
            or np.isclose(best_dir[0], np.pi / 2.0)
            else (points2[0], points1[-1])
        )
        return WaveguidePath(
            self,
            points1 + [np.array([coords1[0], coords2[1]])] + points2,
            geometry,
            r_vals,
            input_geometry=port1.geometry,
            output_geometry=port2.geometry,
        )

    def user_route(
        self,
        geometry,
        r_vals,
        inputs=[],
        outputs=[],
        bbox=None,
        route_file=None,
        current_gds=None,
        port_geometries=[],
    ):
        """
        Returns a user specified route that has been modified to fit the given specifications. Opens a routing GUI if necessary to allow for user input.

        Parameters
        ----------
        geometry : WaveguideGeometry
            Geometry of waveguides.
        r_vals : list of floats
            Possible bend radii in GDS units.
        inputs : list[Port]
            List of input ports.
        outputs : list[Port]
            List of output ports such that each input port corresponds to the output port at the same index.
        route_file : str
            Path to a .route file with the desired initial paths.
        current_gds : str
            Path to the GDS for which the routes are being generated.
        port_geometries : list[WaveguideGeometry]
            Additional geometries of ports if they differ from the routing waveguide geometry.

        Returns
        -------
        paths : WaveguidePath
            WaveguidePaths between each input and output port.
        """

        if len(inputs) != len(outputs):
            raise ValueError("Input and output port arrays must have the same length")

        if len(r_vals) == 0 or min(r_vals) <= 0:
            raise ValueError("Minimum radius must be positive")

        inputs, outputs, initial_paths = self.get_rough_paths(
            inputs=inputs,
            outputs=outputs,
            bbox=bbox,
            geometries=[geometry]+port_geometries,
            route_file=route_file,
            current_gds=current_gds,
        )
        if initial_paths is None:
            raise ValueError("No initial paths specified")

        final_paths = []
        for i in range(len(inputs)):
            path = initial_paths[i]
            if len(path) == 2:
                final_paths.append(
                    WaveguidePath(
                        self,
                        path,
                        geometry,
                        r_vals,
                        input_geometry=inputs[i].geometry,
                        output_geometry=outputs[i].geometry,
                    )
                )
                continue
            curr_dir = utils.manhattan_angle(utils.horizontal_angle(path[1] - path[0]))
            manhattan_path = self.turn_port_route(inputs[i], min(r_vals), curr_dir)
            for j in range(1, len(path) - 2):
                manhattan_path.append(
                    utils.ray_project(manhattan_path[-1], curr_dir, path[j])
                )
                curr_dir = utils.manhattan_angle(
                    utils.horizontal_angle(path[j + 1] - path[j])
                )
            final_dir = utils.manhattan_angle(
                utils.horizontal_angle(path[-1] - path[-2])
            )
            output_turn = self.turn_port_route(
                outputs[i], min(r_vals), final_dir, reverse=True
            )
            intermediate = utils.ray_intersect(
                manhattan_path[-1],
                curr_dir,
                output_turn[0],
                utils.reverse_angle(final_dir),
            )
            if intermediate is None:
                raise ValueError("Provided path is not a valid Manhattan route")
            manhattan_path.extend([intermediate, *output_turn])
            final_paths.append(
                WaveguidePath(
                    self,
                    manhattan_path,
                    geometry,
                    r_vals,
                    input_geometry=inputs[i].geometry,
                    output_geometry=outputs[i].geometry,
                )
            )
        return final_paths

    def write_paths_to_gds(
        self,
        paths,
        output_file,
        layer=0,
        datatype=0,
        layer2=1,
        datatype2=0,
        style="segmented",
    ):
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
        layer2 : int
            Desired GDS layer for partial etch.
        datatype2 : int
            Datatype label for partial etch GDS layer.
        style: 'segmented' or 'continuous'
            'segmented' separates bends from straights, while 'continuous' creates a single polygon that is split up arbitrarily if the vertex count grows too large.

        Returns
        -------
        paths : list[WaveguidePath]
            List of WaveguidePaths between their corresponding inputs and outputs.
        """
        lib = gdstk.Library()
        cell = lib.new_cell("AUTOROUTE")
        for path in paths:
            points = path.points
            N = len(points)
            prev_tangent_len = 0
            curr_point = points[0]
            gds_segments = []
            gds_bends = []
            for i in range(N - 1):
                segment_len = path.get_length(i)
                if i == 0 and path.input_geometry != path.wg_geometry:
                    if path.input_taper_poly1 is not None:
                        p = gdstk.Polygon(
                            path.input_taper_poly1, layer=layer, datatype=datatype
                        )
                        cell.add(p)
                    if path.input_taper_poly2 is not None:
                        p = gdstk.Polygon(
                            path.input_taper_poly2, layer=layer2, datatype=datatype
                        )
                        cell.add(p)
                    prev_tangent_len = path.input_taper_len
                    curr_point = path.input_taper_end
                if i == N - 2:
                    if path.wg_geometry != path.output_geometry:
                        if path.output_taper_poly1 is not None:
                            p = gdstk.Polygon(
                                path.output_taper_poly1, layer=layer, datatype=datatype
                            )
                            cell.add(p)
                        if path.output_taper_poly2 is not None:
                            p = gdstk.Polygon(
                                path.output_taper_poly2, layer=layer2, datatype=datatype
                            )
                            cell.add(p)
                    if not np.isclose(utils.euclidean_distance(curr_point, path.output_taper_end), 0, atol=1e-3):
                        gds_segments.append((curr_point, path.output_taper_end))
                else:
                    angle = utils.path_angle(points[i], points[i + 1], points[i + 2])
                    interior_angle = np.pi - np.abs(angle)
                    new_tangent_len = (
                        0
                        if np.isclose(interior_angle, 0.0)
                        else path.bend_radii[i + 1] / np.tan(interior_angle / 2)
                    )
                    horiz_angle = utils.horizontal_angle(points[i + 1] - points[i])

                    
                    prev_point = np.copy(curr_point)
                    curr_point += (
                        segment_len - prev_tangent_len - new_tangent_len
                    ) * np.array([np.cos(horiz_angle), np.sin(horiz_angle)])
                    if not np.isclose(utils.euclidean_distance(prev_point, curr_point), 0, atol=1e-3):
                        gds_segments.append((prev_point, curr_point))

                    start_angle = horiz_angle - np.pi / 2
                    if angle < 0:
                        start_angle = utils.reverse_angle(start_angle)
                    gds_bends.append(
                        (
                            curr_point,
                            path.bend_radii[i + 1],
                            start_angle,
                            start_angle + angle,
                        )
                    )
                    curr_point = (
                        points[i + 1]
                        + (points[i + 2] - points[i + 1])
                        / path.get_length(i + 1)
                        * new_tangent_len
                    )
                    prev_tangent_len = new_tangent_len
            to_draw = [(path.wg_geometry.width, layer, datatype)]
            if path.wg_geometry.kind == "ridge":
                to_draw.append((path.wg_geometry.ridge_width, layer2, datatype2))
            for width, curr_layer, curr_datatype in to_draw:
                if style == "segmented":
                    for segment in gds_segments:
                        rp = gdstk.RobustPath(
                            segment[0], width, layer=curr_layer, datatype=curr_datatype
                        )
                        rp.segment(segment[1])
                        cell.add(rp)
                    for bend in gds_bends:
                        rp = gdstk.RobustPath(
                            bend[0],
                            width,
                            layer=curr_layer,
                            datatype=curr_datatype,
                        )
                        rp.arc(*bend[1:])
                        cell.add(rp)
                elif style == "continuous":
                    rp = gdstk.RobustPath(
                        gds_segments[0][0],
                        width,
                        layer=curr_layer,
                        datatype=curr_datatype,
                    )
                    for i in range(len(gds_segments) - 1):
                        rp.segment(gds_segments[i][1])
                        rp.arc(*gds_bends[i][1:])
                    rp.segment(gds_segments[-1][1])
                    cell.add(rp)
                else:
                    raise ValueError("Invalid style for GDS export")

        lib.write_gds(output_file)
