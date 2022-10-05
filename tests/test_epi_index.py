# -*- coding: utf-8 -*-


import numpy as np

import seeg


def test_find_onsets(synthetic):
    ER = seeg.calc_ER(synthetic)
    U_n = seeg.cusum(ER)
    # print(U_n.shape)
    onsets = seeg.find_onsets(U_n, synthetic.ch_names, 1, 0.25, 5)
    np.testing.assert_allclose(onsets['detection_time'], [np.nan, 2, 2.5])
    np.testing.assert_allclose(onsets['alarm_time'], [np.nan, 2.5, 3.])


def test_calculate_EI(synthetic):
    onsets = seeg.calculate_EI(synthetic)
    np.testing.assert_allclose(onsets['EI'], [0, 1., 0.665],
                               rtol=1e-3)
