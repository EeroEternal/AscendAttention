"""Compiled kernels, one subpackage per family.

Nothing here is imported eagerly: every kernel module pulls in the TileLang toolchain (guarded at
module scope where it must be), and ``import ascend_attention`` has to stay toolchain-free.
"""
