# -*- coding: utf-8 -*-


import numpy as np

import seeg


def test_find_onsets(raw):
    ER = seeg.calc_ER(raw)
    U_n = seeg.cusum(ER)
    print(U_n.shape)
    onsets = seeg.find_onsets(U_n, raw.ch_names)
    np.testing.assert_allclose(onsets['detection_time'], [np.nan, 2, 2.5])
    np.testing.assert_allclose(onsets['alarm_time'], [np.nan, 2.5, 3.])


def test_calculate_EI(raw):
    onsets = seeg.calculate_EI(raw)
    np.testing.assert_allclose(onsets['EI'], [0, 1., 0.667],
                               rtol=1e-3)
