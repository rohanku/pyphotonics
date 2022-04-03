from pyphotonics.simulation import lumerical, sources, structure
import numpy as np
import os, string, random

lumapi = lumerical.lumapi


def _setup_rr_mode(mode, wavelength):
    mode.addrect(
        name="add_wg",
        x=0,
        x_span=16e-6,
        y=0,
        y_span=16e-6,
        z=-2e-6,
        z_span=4e-6,
        material="SiO2 (Glass) - Palik",
    )

    rr = mode.addobject(
        "ring_resonator",
        x=-7e-6,
        y=0,
        z=0.09e-6,
        lc=0,
        gap=0.1e-6,
        radius=4e-6,
        material="Si (Silicon) - Palik",
        base_width=0.4e-6,
        base_height=0.18e-6,
        x_span=14e-6,
        base_angle=90,
    )

    fde = mode.addfde(
        solver_type="2D X normal",
        x=-5e-6,
        y=3.6e-6,
        y_span=4e-6,
        z=0.075e-6,
        z_span=1e-6,
        mesh_cells_z=100,
        mesh_cells_y=200,
        wavelength=1.5e-6,
        track_selected_mode=1,
        stop_wavelength=1.6e-6,
        number_of_points=4,
        detailed_dispersion_calculation=1,
    )


def ring_resonator_char(
    wavelength,
    interactive=False,
    sim_time=2000e-15,
    io_buffer=3,
    mesh_buffer=None,
):
    print("Starting Lumerical session...")
    with lumapi.MODE() as mode:
        print("Setting up simulation...")
        _setup_rr_mode(mode, wavelength)

        # Create temporary simulation folder and wait for user input before running if interactive
        tmp_dir = "/tmp/pyphotonics/ring_resonator_char"
        os.makedirs(tmp_dir, exist_ok=True)
        tag = "".join(random.choice(string.ascii_letters) for i in range(10))
        fname = f"{tmp_dir}/rr_{tag}.lms"
        print(f"Saving simulation file as {fname}...")
        mode.save(fname)

        if interactive:
            i = input("Press Enter to continue (or q to quit)...")
            if i == "q":
                quit()

        mode.run()

        mode.findmodes()
        mode.selectmode(1)
        mode.frequencysweep()

        R = mode.getresult("frequencysweep")
        T = mode.getresult("frequencysweep", "f")

        print("Simulation complete, returning results.")


if __name__ == "__main__":
    ring_resonator_char(1.5e-6, interactive=False)
