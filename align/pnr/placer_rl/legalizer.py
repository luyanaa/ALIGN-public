"""Shared, torch-free placement legaliser for the RALF-style placers.

The SA path (``sa_solver``) and the RL path (``rl_solver``) both produce a
``{inst_idx: (x, y, orient)}`` placement that may violate the module's hard
constraints: symmetric blocks are only *approximately* mirrored and the RL
grid scatter overlaps heavily.  This module owns the post-processing that
turns such a placement into a legal floorplan:

* snap symmetric pairs onto their group axis **as units** (IdeaPlaceEx's
  ``GridAligner::adjustSymPair`` rule: snap one member to the grid, then
  derive the partner's position from the group's axis), centring
  self-symmetric blocks;
* pair-locked de-overlap (symmetric partners translate together so their
  mirror relationship survives; self-symmetric blocks move alone);
* a final axis re-snap so the hard mirror equality holds exactly after the
  de-overlap pass.

It is pure Python (no torch) so the SA path stays dependency-free.  The
per-group axis derivation reuses :func:`cost_model.group_axis` /
:func:`cost_model.symmetry_axes`; the extra ``feasible`` variant additionally
raises the median centreline so the widest centred block still fits at a
non-negative coordinate, matching the SA solver's historical behaviour.
"""

from .cost_model import (_strict_intersection_area, group_axis,
                         symmetry_axes)

__all__ = [
    'group_axis', 'symmetry_axes',
    'median_centerline', 'feasible_axis',
    'sym_partners', 'snap_group',
    'deoverlap', 'legalize',
]


def median_centerline(placement, problem, group, direction=None):
    """Median member centreline of ``group`` along its mirror direction."""
    direction = direction or group.get('direction', 'V')
    insts = problem.instances
    centers = []
    for pair in group['pairs']:
        for i in pair:
            x, y, _ = placement[i]
            if direction == 'V':
                centers.append(x + insts[i]['width'] / 2)
            else:
                centers.append(y + insts[i]['height'] / 2)
    centers.sort()
    return centers[len(centers) // 2]


def feasible_axis(placement, problem, group):
    """Median centreline, raised so the widest centred block still fits with
    llx (or lly) >= 0."""
    direction = group.get('direction', 'V')
    axis = median_centerline(placement, problem, group, direction)
    insts = problem.instances
    for pair in group['pairs']:
        for i in pair:
            extent = (insts[i]['width'] if direction == 'V'
                      else insts[i]['height'])
            axis = max(axis, extent / 2)
    return axis


def sym_partners(groups):
    """Map ``inst_idx -> partner inst_idx`` for two-element pairs."""
    partner = {}
    for group in (groups or []):
        for pair in group['pairs']:
            if len(pair) == 2:
                partner[pair[0]] = pair[1]
                partner[pair[1]] = pair[0]
    return partner


def snap_group(problem, placement, group, axis):
    """Snap one symmetry group's pairs/self-syms to ``axis`` as units.

    Mutates ``placement`` in place.  Assumes matched-size pairs, as ALIGN's
    ``SymmetricBlocks`` pairs normally are (IdeaPlaceEx makes the same
    assumption).
    """
    insts = problem.instances
    direction = group.get('direction', 'V')
    pl = placement
    for pair in group['pairs']:
        if len(pair) == 1:
            i = pair[0]
            x, y, o = pl[i]
            if direction == 'V':
                w = insts[i]['width']
                pl[i] = (int(round(axis - w / 2)), y, o)
            else:
                h = insts[i]['height']
                pl[i] = (x, int(round(axis - h / 2)), o)
        elif len(pair) == 2:
            i0, i1 = pair
            if direction == 'V':
                w0, w1 = insts[i0]['width'], insts[i1]['width']
                # pick the left member (adjustSymPair does the same)
                if pl[i0][0] <= pl[i1][0]:
                    left, right, wl = i0, i1, w0
                else:
                    left, right, wl = i1, i0, w1
                xl, yl, ol = pl[left]
                xr, yr, orr = pl[right]
                # snap the left member to the grid, derive the partner
                xl = int(round(xl))
                xr = int(round(2 * axis - xl - wl))
                # keep the shared y-centre (ALIGN aligns pair centres)
                yc = (yl + insts[left]['height'] / 2
                      + yr + insts[right]['height'] / 2) / 2
                pl[left] = (xl,
                            int(round(yc - insts[left]['height'] / 2)), ol)
                pl[right] = (xr,
                             int(round(yc - insts[right]['height'] / 2)),
                             orr)
            else:
                h0, h1 = insts[i0]['height'], insts[i1]['height']
                if pl[i0][1] <= pl[i1][1]:
                    bottom, top, hb = i0, i1, h0
                else:
                    bottom, top, hb = i1, i0, h1
                xb, yb, ob = pl[bottom]
                xt, yt, ot = pl[top]
                yb = int(round(yb))
                yt = int(round(2 * axis - yb - hb))
                xc = (xb + insts[bottom]['width'] / 2
                      + xt + insts[top]['width'] / 2) / 2
                pl[bottom] = (int(round(xc - insts[bottom]['width'] / 2)),
                              yb, ob)
                pl[top] = (int(round(xc - insts[top]['width'] / 2)),
                           yt, ot)


def deoverlap(placement, problem, groups=None):
    """Greedy horizontal push to remove bbox overlaps (keeps y).

    When ``groups`` is given, symmetric partners are translated together by
    the same offset so their mirror relationship survives; a pair that
    overlaps *itself* cannot be fixed by joint translation and is left for
    the final axis snap.  Self-symmetric blocks move alone and are re-centred
    afterwards by :func:`legalize`.
    """
    insts = problem.instances
    partner = sym_partners(groups)
    pl = {i: list(v) for i, v in placement.items()}
    for _ in range(200):
        moved = False
        items = sorted(pl.items(), key=lambda kv: kv[1][0])
        boxes = {i: (v[0], v[1], v[0] + insts[i]['width'],
                     v[1] + insts[i]['height']) for i, v in pl.items()}
        for a in range(len(items)):
            for b in range(a + 1, len(items)):
                i, j = items[a][0], items[b][0]
                ov = _strict_intersection_area(boxes[i], boxes[j])
                if ov > 0:
                    if partner.get(j) == i:
                        # mutually symmetric overlapping pair: joint
                        # translation cannot separate it
                        continue
                    # push the right-hand block right past the left one
                    shift = boxes[i][2] - boxes[j][0] + 1
                    moved_ids = [j]
                    if j in partner:
                        moved_ids.append(partner[j])
                    for k in moved_ids:
                        pl[k][0] += shift
                        boxes[k] = (pl[k][0], pl[k][1],
                                    pl[k][0] + insts[k]['width'],
                                    pl[k][1] + insts[k]['height'])
                    moved = True
        if not moved:
            break
    return {i: tuple(v) for i, v in pl.items()}


def legalize(problem, placement, groups=None):
    """Legalise ``placement`` against ``problem``'s symmetry constraints.

    Returns ``(legalised_placement, sym_axes)`` where ``sym_axes`` is the
    per-group feasible axis derived from the *input* placement (one entry per
    group, aligned with ``problem.symmetry``).  When the problem (or the
    supplied ``groups``) has no symmetry this reduces to a plain
    :func:`deoverlap` and returns ``axes == []``.

    ``placement`` is a ``{inst_idx: (x, y, orient)}`` dict; the returned dict
    has the same shape.
    """
    groups = problem.symmetry if groups is None else groups
    pl = {i: tuple(v) for i, v in placement.items()}
    if not groups:
        return deoverlap(pl, problem), []
    axes = [feasible_axis(pl, problem, group) for group in groups]
    for group, axis in zip(groups, axes):
        snap_group(problem, pl, group, axis)
    pl = deoverlap(pl, problem, groups)
    # De-overlap moves pairs together and may displace a self-symmetric
    # block; re-snap to the (unchanged) axes so the hard equality holds
    # exactly after legalisation.
    for group, axis in zip(groups, axes):
        snap_group(problem, pl, group, axis)
    return pl, list(axes)
