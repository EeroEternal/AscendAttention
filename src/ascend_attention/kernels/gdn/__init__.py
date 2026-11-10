"""GDN kernels: the fused decode conv update and the solve_tril+recompute_w_u fusion.

Both import the TileLang dialect at module scope (guarded), so nothing here is imported eagerly.
"""
