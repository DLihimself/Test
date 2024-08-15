import simulator.data_simulator as dtsm


class InferenceModule():
    def __init__(self):
        self.smfdt = self.get_feature_dist()

    def get_feature_dist(self):
        n_samples = 1000
        fdt = dict.fromkeys(dtsm.FEATURE_COL, None)
        for fname in fdt:
            fdt[fname] = dict.fromkeys([dtsm.NORM, dtsm.ABNR], None)

        for fname, fdt_dict in fdt.items():
            smfdt_norm, _ = dtsm.simulate_feature_value(fname, n_samples, flabel=dtsm.NORM)
            fdt_dict[dtsm.NORM] = smfdt_norm

            smfdt_abnr, _ = dtsm.simulate_feature_value(fname, n_samples, flabel=dtsm.ABNR)
            fdt_dict[dtsm.ABNR] = smfdt_abnr

        return fdt

    def is_abnormal_feature(self, fname, fvalue, tol=1e-3):
        fdt_norm = self.smfdt[fname][dtsm.NORM]
        fdt_abnr = self.smfdt[fname][dtsm.ABNR]
        fscale = fdt_norm.kwds['scale']
        fbin = fscale/100.0
        fbin_low = fvalue - fbin
        fbin_up = fvalue + fbin
        
        p_norm = fdt_norm.cdf(fbin_up) - fdt_norm.cdf(fbin_low)
        p_abnr = fdt_abnr.cdf(fbin_up) - fdt_abnr.cdf(fbin_low)

        result = True if (p_norm-p_abnr) < 1e-3 else False
        return result

    def get_feature_label(self, signal_data):
        n_features = len(dtsm.FEATURE_COL)
        fmask = [False] * n_features
        for i in range(n_features):
            fname = dtsm.FEATURE_COL[i]
            fvalue = signal_data[fname]
            fmask[i] = self.is_abnormal_feature(fname, fvalue)

        return fmask

    def set_abnormal_feature_nan(self, signal_data):
        fmask = self.get_feature_label(signal_data)
        signal_data[fmask] = np.nan
        return signal_data





