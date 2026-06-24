# Gravitational Wave Detector Sensitivities (PLISC)

**For paper figures**, use the **published** Schmitz tables in `analysis/data/plis_*.dat`
(Zenodo [10.5281/zenodo.3689582](https://doi.org/10.5281/zenodo.3689582),
`power-law-integrated_sensitivities.tar.gz`). Loaded by `analysis/published_plisc.py`
and `analysis/computeGW_gamma.py --plot`.

The CSV files in `data/gw_detectors/` were generated locally from noise PSDs via
`analysis/export_detector_data.py` (computed PLISC envelope). They are **not** the
same tabulated curves as Schmitz (2020) and should not be used for publication plots.

---

The computed PLISC in `data/gw_detectors/` was generated from the raw Noise Power
Spectral Density ($S_h(f)$) for each detector, converted into PLISC form.

The PLISC represents the envelope of power-law signal amplitudes that would produce a Signal-to-Noise Ratio (SNR) of exactly 1 after an observation time of 1 year. This is the standard method used by Schmitz (2020) [arXiv:2002.04615] and Thrane & Romano (2013) to plot detector sensitivity curves against stochastic GW backgrounds.

The raw noise equations used before applying the PLISC conversion are:

- **LISA**: From Caprini et al. (2016) [arXiv:1512.06239]
- **DECIGO / BBO**: From Yagi & Seto (2011) [arXiv:1101.3940], BBO is approx DECIGO / 100
- **Einstein Telescope (ET)**: From ET Design Study [arXiv:1101.3940]
- **Advanced LIGO (aLIGO)**: Design sensitivity, from Thrane & Romano (2013) [arXiv:1310.5300]
