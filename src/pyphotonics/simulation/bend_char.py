import lumerical
import modes
import structure
import numpy as np
import os, string, random
import matplotlib.pyplot as plt
from scipy.constants import c

lumapi = lumerical.lumapi

def soi_characterize_bend_varfdtd(
    soi,
    width,
    radius,
    angle,
    wavelength,
    interactive=False,
    io_buffer=10,
    mesh_buffer=None,
):
    """
    Returns the transmission of an SOI slab waveguide bend of a given width, radius, and angle at a given wavelength.

    Parameters:
        soi (structure.SOI):
            SOI object with the desired parameters

        width (double):
            Waveguide width in meters

        radius (double):
            Bend radius in meters

        angle (double):
            Bend angle in degrees

        wavelength (double):
            Target wavelength in meters

        interactive (bool):
            Interactive mode opens the created simulation for inspection before running

        io_buffer (double):
            Spacing between bend and the input source/output monitor relative to wavelength

        mesh_buffer (double):
            Buffer around input waveguide and bend for meshes to span in meters. Defaults to no mesh.

    Returns:
        T (double):
            Bend transmission as measured at output monitor
    """
    # Validate radius
    if radius < width / 2:
        raise ValueError("Radius must be at least half of the width of the waveguide")

    # Validate angle
    if not 0 <= angle <= 180:
        raise ValueError("Angle must be between 0 and 180.")

    print("Starting Lumerical session...")
    with lumapi.MODE(hide=not interactive) as mode:
        print("Setting up simulation...")

        # Compute necessary dimensions
        angle_rad = np.radians(angle)

        sim_x_min = 1 - radius * np.cos(angle_rad) - 2 * io_buffer * wavelength * np.sin(angle_rad) - width * np.cos(angle_rad)
        sim_y_min = -2 * io_buffer * wavelength
        sim_x_max = io_buffer * wavelength + radius
        sim_y_max = radius * np.sin(angle_rad) + 2 * io_buffer * wavelength * np.cos(angle_rad) + width * np.sin(angle_rad)
        sim_diag = ((sim_x_max - sim_x_min)**2 + (sim_y_max - sim_y_min)**2)**(1/2)

        sim_time = (2*sim_diag + radius * angle_rad)/c

        input_wg_len = sim_diag
        input_wg_x = radius * np.cos(angle_rad) - input_wg_len * np.sin(angle_rad) / 2
        input_wg_y = radius * np.sin(angle_rad) + input_wg_len * np.cos(angle_rad) / 2

        # Set up mode source on input waveguide
        source_x = radius * np.cos(angle_rad) - (
            io_buffer
        ) * wavelength * np.sin(angle_rad)
        source_y = radius * np.sin(angle_rad) + (
            io_buffer
        ) * wavelength * np.cos(angle_rad)

        # Set up simulation
        varfdtd = mode.addvarfdtd(
            x_min=sim_x_min,
            x_max=sim_x_max,
            y_min=sim_y_min,
            y_max=sim_y_max,
            z=soi.si_t / 2,
            z_span=10 * soi.si_t,
            x0=radius,
            y0=-radius / 2,
            simulation_wavelength_min=wavelength,
            simulation_wavelength_max=wavelength,
            simulation_time=sim_time,
        )

        # Set up substrate, BOX, device layer, and TOX
        soi.addsoi(mode)

        # If the angle is nonzero, we need to add a bend
        if angle != 0:
            # Create bend centered at the origin with the desired radius and angle
            mode.addring(
                name="bend",
                x=0,
                y=0,
                z_min=0,
                z_max=soi.si_t,
                inner_radius=radius - width / 2,
                outer_radius=radius + width / 2,
                theta_start=0,
                theta_stop=angle,
                material="Si (Silicon) - Palik",
            )
            # Add mesh to bend
            if mesh_buffer != None:
                mode.addmesh(
                    name="bend_mesh",
                    based_on_a_structure=2,
                    structure="bend",
                    buffer=mesh_buffer,
                )

        # Create output waveguide parallel to the y-axis
        mode.addrect(
            name="output_wg",
            x=radius,
            x_span=width,
            y_min=sim_y_min - 1e-6,
            y_max=0,
            z_min=0,
            z_max=soi.si_t,
            material="Si (Silicon) - Palik",
        )

        # Create angled input waveguide
        mode.addrect(
            name="input_wg",
            x=input_wg_x,
            x_span=width,
            y=input_wg_y,
            y_span=input_wg_len,
            z_min=0,
            z_max=soi.si_t,
            first_axis="z",
            rotation_1=angle,
            material="Si (Silicon) - Palik",
        )
        # Add mesh to angled waveguide
        if mesh_buffer != None:
            mode.addmesh(
                name="input_mesh",
                based_on_a_structure=1,
                structure="input_wg",
                buffer=mesh_buffer,
            )

        angle_dir = (
            45 <= angle <= 135
        )  # Mode sources don't work well for oblique angles, so we need to place it in the closest possible direction
        if angle_dir:
            mode.addmodesource(
                name="input_source",
                injection_axis=1,
                direction=2,
                theta=angle - 90,
                x=source_x,
                y=source_y,
                y_span=2 * width,
                center_wavelength=wavelength,
                wavelength_span=0,
                mode_selection="user select",
            )
        else:
            mode.addmodesource(
                name="input_source",
                injection_axis=2,
                direction=1,
                theta=angle,
                x=source_x,
                x_span=2 * width,
                y=source_y,
                center_wavelength=wavelength,
                wavelength_span=0,
                mode_selection="user select",
            )

        # Calculate fundamental TE mode of input waveguide and update mode source
        mode_num = modes.get_fundamental_te_mode(mode)
        print(mode_num)
        if mode_num == -1:
            raise ValueError("Failed to find fundamental TE mode for given parameters")
        mode.updatesourcemode(mode_num)

        # Add longitudinal monitor for qualitative characterization purposes
        mode.addpower(
            name="longitudinal_monitor",
            monitor_type=7,
            x_min=sim_x_min,
            x_max=sim_x_max,
            y_min=sim_y_min,
            y_max=sim_y_max,
            z=soi.si_t / 2,
            down_sample_x=3,
            down_sample_y=3,
        )

        # Add output monitor for power and transmission data
        mode.addpower(
            name="output_monitor",
            monitor_type=6,
            x=radius,
            x_span=2 * width,
            y=-wavelength * io_buffer,
            z=soi.si_t / 2,
            z_span=10 * soi.si_t,
        )

        print("Saving and running simulation...")
        # Create temporary simulation folder and wait for user input before running if interactive
        os.makedirs("/tmp/optics_lib/bend_char", exist_ok=True)
        tag = "".join(random.choice(string.ascii_letters) for i in range(10))
        mode.save(
            f"/tmp/optics_lib/bend_char/width_{int(width*1e9)}nm_radius_{int(radius*1e9)}nm_angle_{int(angle)}_{tag}.lms"
        )
        if interactive:
            input("Press Enter to continue...")
        mode.run()

        print("Simulation complete, returning results.")
        # Return the simulated T value
        T = mode.getresult("output_monitor", "T")
        return -T["T"][0]


if __name__ == "__main__":
    for angle in range(0, 100, 10):
        print(characterize_bend_varfdtd(500e-9, 1.5e-6, angle, interactive=False))
