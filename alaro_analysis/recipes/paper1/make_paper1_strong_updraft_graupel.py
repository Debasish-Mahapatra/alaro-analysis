"""Paper1 graupel profile, using an explicit immutable variable specification."""

from alaro_analysis.common.study_specs import GRAUPEL_PROFILE
from alaro_analysis.recipes.paper1.make_paper1_strong_updraft_rain import (
    main as render_profile,
)


def main(argv=None):
    return render_profile(argv, spec=GRAUPEL_PROFILE)


if __name__ == "__main__":
    main()
