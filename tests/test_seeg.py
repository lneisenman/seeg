# -*- coding: utf-8 -*-

"""
test_seeg
----------------------------------

Tests for `seeg` module.
"""

from __future__ import (print_function, division, absolute_import,
                        unicode_literals)

import pytest


from seeg import seeg_fcn


def test_seeg():
    assert seeg_fcn()   # this should pass
