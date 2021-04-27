# -*- coding: utf-8 -*-

import numpy as np

import seeg


def test_line_length(raw):
    ll, sd = seeg.line_length(raw)
    assert ll.shape[-1] == 37
    np.testing.assert_allclose(sd, [2.7381e-6]*3, rtol=1e-3)


def test_llei(raw):
    onsets = seeg.line_length_EI(raw)
    print(onsets)
    assert np.abs(onsets.loc[2, 'LLEI'] - 0.6667) < 0.01
