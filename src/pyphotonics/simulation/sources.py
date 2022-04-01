def get_fundamental_te_mode(sim):
    """
    Retrieves the number of the fundamental TE mode of the mode source selected in the given Lumerical simulation.

    Parameters
    ----------
    sim: Lumerical API simulation object
        Simulation with mode source of interest selected.

    Returns
    -------
    mode_num : int
        Fundamental TE mode number. Returns -1 if a TE mode could not be found.
    """

    # Store original mode source name
    mode_source = sim.get("name")

    # Copy selected mode source
    sim.copy()
    tmp_source = "tmp_source"
    sim.set("name", tmp_source)

    # Check
    for i in range(1, 11):
        sim.updatesourcemode(i)
        if (
            sim.getresult(tmp_source, "TE polarization fraction")[
                "TE polarization fraction"
            ][0]
            > 0.5
        ):
            sim.delete()
            sim.select(mode_source)
            return i
    sim.delete()
    sim.select(mode_source)
    return 1
