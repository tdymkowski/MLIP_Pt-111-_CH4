from xpot.models import PACE
from xpot.optimiser import NamedOptimiser

mlip = PACE("MLIP_input.hjson")

print(mlip.optimisation_space)

kwargs = {
        "n_initial_points": 5
        }

print(mlip.ace_total)
opt = NamedOptimiser(mlip.optimisation_space, mlip.sweep_path, kwargs)

n_calls = 10

while opt.iter <= n_calls:
    opt.run_optimisation(mlip.fit, path = mlip.sweep_path)

opt.tabulate_final_results(mlip.sweep_path)
