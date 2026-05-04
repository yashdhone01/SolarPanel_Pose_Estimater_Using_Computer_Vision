def check_hazard(mean_residual):
    """
    Evaluates if the fitted plane residual implies a rugged/bumpy surface.
    Returns True if HAZARD, False if SAFE.
    """
    # Plane fit returns mean_residual in the same scale as Z (meters).
    # If the average deviation from the plane > 0.05 (5cm), surface is rugged.
    if mean_residual is None:
        return True
    return mean_residual > 0.05
