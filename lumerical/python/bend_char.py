import lumerical
import structure
import numpy as np

lumapi = lumerical.get_api()


def characterize_bend_varfdtd(width, radius, angle):
    '''
    Returns the transmission for an SOI slab waveguide of the given width (m), radius (m), and angle (deg).
    '''
    angle_rad = np.radians(angle)

    # Validate radius
    if radius < width / 2:
        raise TypeError("Radius must be at least half of the width of the waveguide")

    # Validate angle
    if not 0 <= angle <= 180:
        raise TypeError("Angle must be between 0 and 180.")

    with lumapi.MODE() as mode:
        help(mode.addvarfdtd)
        varfdtd = mode.addvarfdtd(x=0, x_span=4*radius, y=0, y_span=4*radius, z=structure.si_t/2, z_span=10*structure.si_t, x0=radius, y0=-radius/2)
        structure.addsoi(mode)
        mode.addring(
            name="bend",
            x=0,
            y=0,
            z_min=0,
            z_max=structure.si_t,
            inner_radius=radius - width / 2,
            outer_radius=radius + width / 2,
            theta_start=0,
            theta_stop=angle,
            material="Si (Silicon) - Palik",
        )
        mode.addrect(
            name="output_wg",
            x=radius,
            x_span=width,
            y_min=-5*radius,
            y_max=0,
            z_min=0,
            z_max=structure.si_t,
            material="Si (Silicon) - Palik",
        )
        input_wg_x = radius*(np.cos(angle_rad) - 5*np.sin(angle_rad)/2)
        input_wg_y = radius*(np.sin(angle_rad) + 5*np.cos(angle_rad)/2)
        mode.addrect(
            name="input_wg",
            x=input_wg_x,
            x_span=width,
            y=input_wg_y,
            y_span=5*radius,
            z_min=0,
            z_max=structure.si_t,
            first_axis="z",
            rotation_1=angle,
            material="Si (Silicon) - Palik",
        )
        source_x = radius*(np.cos(angle_rad) - np.sin(angle_rad)/2)
        source_y = radius*(np.sin(angle_rad) + np.cos(angle_rad)/2)
        mode.addmodesource(injection_axis=2, theta=angle, x=source_x, x_span=2*width, y=source_y, center_wavelength=structure.wavelength, wavelength_span=0) # Make backwards
        mode.addpower(monitor_type=6, x=radius, x_span=2*width, y=-radius/2, z=structure.si_t/2, z_span=10*structure.si_t)
        input("Press Enter to continue...")


if __name__ == "__main__":
    characterize_bend_varfdtd(440e-9, 1.5e-6, 30)
