substrate_s = 1.5e-3
substrate_t = 725e-6
box_t = 2e-6
si_t = 220e-9
tox_t = 1e-6


def addsoi(sim):
    """
    Set up SOI material stack in provided simulation and returns the corresponding layers. Takes the simulation object as a parameter.
    """

    # Create substrate
    substrate = sim.addrect(
        name="substrate",
        x=0,
        x_span=substrate_s,
        y=0,
        y_span=substrate_s,
        z_min=-substrate_t - box_t,
        z_max=-box_t,
        material="Si (Silicon) - Palik",
    )

    # Create BOX
    box = sim.addrect(
        name="BOX",
        x=0,
        x_span=substrate_s,
        y=0,
        y_span=substrate_s,
        z_min=-box_t,
        z_max=0,
        material="SiO2 (Glass) - Palik",
    )

    # Create Si etch layer
    si_etch = sim.addrect(
        name="Si_etch",
        x=0,
        x_span=substrate_s,
        y=0,
        y_span=substrate_s,
        z_min=0,
        z_max=si_t,
        material="SiO2 (Glass) - Palik",
    )

    # Create TOX
    tox = sim.addrect(
        name="TOX",
        x=0,
        x_span=substrate_s,
        y=0,
        y_span=substrate_s,
        z_min=si_t,
        z_max=si_t + tox_t,
        material="SiO2 (Glass) - Palik",
    )

    return substrate, box, si_etch, tox
