# G1 GPU5 input preflight — 2026-10-20

The registered GPU5 G1 tracking200 plan was checked in the actual GroundJEV
runtime without allocating a GPU. It has the same 112 registered native rows,
56 two-image pairs, 72 unique images, 884 native proposals, 628 native-aligned
regions, and training schedule as the preserved GPU6 tracking200 reference.

The G1 plan keeps the registered identity-loss coefficients `[0.1, 0.1]` and
physical training GPU 7 in its own plan metadata. No optimizer update was run
and no tracking metric claim is made. The live G1 owner remains gated on the
terminal clean G2 receipt and will still execute the real GPU5 queue command
when that dependency is satisfied.
