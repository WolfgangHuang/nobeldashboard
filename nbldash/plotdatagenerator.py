"""
Facade for the former monolithic plotdatagenerator.py (5.5k lines), now split into:

    data.py             loading / cleaning / enrichment (runs once at import)
    filters.py          shared filter helpers
    figs/geography.py   geography figures
    figs/demography.py  demography figures
    figs/time_age.py    time / age figures
    figs/migration.py   migration figures
    figs/prizemoney.py  prize money figures
    network.py          nominations network + map
    overview_stats.py   overview KPIs

Everything is re-exported here, so app.py uses `pdg.<name>` and PlotConfig looks
up its generator functions by name on this module.

The derived CSV exports are written by pipeline/export_derived.py.
"""

from nbldash.data import *              # noqa: F401,F403
from nbldash.filters import *           # noqa: F401,F403
from nbldash.figs.geography import *    # noqa: F401,F403
from nbldash.figs.demography import *   # noqa: F401,F403
from nbldash.figs.time_age import *     # noqa: F401,F403
from nbldash.figs.migration import *    # noqa: F401,F403
from nbldash.figs.prizemoney import *   # noqa: F401,F403
from nbldash.network import *           # noqa: F401,F403
from nbldash.overview_stats import *    # noqa: F401,F403
