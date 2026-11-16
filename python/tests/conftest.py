import jax

# HRAP is validated in double precision; must be set before any jax arrays are created.
jax.config.update("jax_enable_x64", True)
