"""The module that held the legacy API up to frc 0.2. Its functions now live
in frc.legacy, and frc imports them all."""

from frc.legacy import frc_res, half_bit_threshold, one_frc, two_frc

__all__ = ['one_frc', 'two_frc', 'frc_res']
