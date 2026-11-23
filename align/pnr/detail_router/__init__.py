"""Modern Python detailed router for ALIGN.

A grid-based, track-aligned detailed router intended as a clean-room
successor to the Intel `AnalogDetailedRouter` idea: nets are routed by
choosing candidate track segments and vias on a per-layer routing grid,
subject to connectivity and design-rule constraints, solved by an
external constraint engine.

Two interchangeable solver backends share the same candidate/constraint
model and only differ in the final ~200 lines of encoding:

* ``solver_sat``  -- boolean SAT via ``python-sat`` (Glucose/CaDiCaL),
  the closest analogue to AMSR's ``solver_type=glucose``.
* ``solver_mip``  -- integer programming via ``mip`` (CBC / HiGHS),
  easier to extend with weighted objectives and multi-core MIP search.

Multi-core speedup comes from *spatial partitioning*: ``partition.py``
groups nets into disjoint congestion regions which are routed
concurrently with ``multiprocessing``.  The per-region solver stays
single-threaded (both SAT and CBC/HiGHS are primarily sequential).

Pipeline:
    scaled_placement_verilog + leaf collateral
      -> grid.py            build the track-aligned routing grid
      -> candidates.py      generate candidate wires/vias per net
      -> constraints.py     encode connectivity + spacing + guide prefs
      -> partition.py       split nets into independent regions
      -> solver_sat/mip     solve each region
      -> driver.py          emit router-output JSON (terminals schema)
"""
