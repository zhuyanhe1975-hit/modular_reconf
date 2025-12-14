import itertools
from .collision import create_box, check_collision


def check_all_collisions(modules):
    """Check for collisions between all pairs of modules."""
    box = create_box(size=0.1)  # Assume uniform box geometry for all modules

    module_ids = list(modules.keys())
    for id1, id2 in itertools.combinations(module_ids, 2):
        m1 = modules[id1]
        m2 = modules[id2]
        if check_collision(box, m1.world_T, box, m2.world_T):
            return True
    return False
