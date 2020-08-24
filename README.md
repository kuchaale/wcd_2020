[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3780102.svg)](https://doi.org/10.5281/zenodo.3780102)
[![Python 3.7](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-369/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

# On the intermittency of orographic gravity wave hotspots and its importance for middle atmosphere dynamics
**A. Kuchar, P. Sacha, R. Eichinger, Ch. Jacobi, P. Pisoft, and H. Rieder**

In discusssion in [Weather and Climate Dynamics](https://www.weather-clim-dynam-discuss.net/wcd-2020-21/).

Code used to process and visualise the model and other data outputs in order to reproduce figures in the manuscript.
Model data are available [here](http://climate-modelling.canada.ca/climatemodeldata/cmam/output/CMAM/CMAM30-SD/index.shtml). All datasets already preprocessed can be found [here](https://data.mendeley.com/datasets/j3hj7f9t67/draft?a=58611508-4e4e-4f44-8e39-c13080528787).

Notebooks for each individual figure as well as for two data tables are in the [`code/` directory](code), while the figures themselves are in the [`plots/` directory](plots).

### Figures
|  #  | Figure                                                                                                                                                                                                    | Notebook                                                                              | Dependencies                                                                                                                                                             |
|:---:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|  1 | [Ratio of zonally averaged OGWD in zonal direction to the total wave forcing](plots/ratio_oGWD_vs_PWD+noGWD_climatology_wSSW_all.pdf)                                                                              | [reproduce_Fig1+2.ipynb](code/reproduce_Fig1+2.ipynb)                       |                                                                                                                                       |
|  2 | [Winter climatology of the zonal OGWD component at 70 hPa](plots/oGWD_map@70hPa_climatology_hotspots_DJF_LCp.pdf)                                                      | [reproduce_Fig1+2.ipynb](code/reproduce_Fig1+2.ipynb)                 |                                                                                                                           |
|  3 | [Area-weighted average of daily OGWD within the hotspot areas](plots/static_ts_plot_ogwd_hotspots@70hPa.pdf)                | [reproduce_Fig3+table.ipynb](code/reproduce_Fig3+table.ipynb)                 | [averaging.py](code/averaging.py), [detect_peaks.py](code/detect_peaks.py)                                                                                                                            |
|  4 | [Climatologies for the absolute gravity wave momentum fluxes at 20 and 30 km](plots/CMAM_vs_GRACILE_DJF_log.pdf) | [reproduce_Fig4.ipynb](code/reproduce_Fig4.ipynb)                     |                                                                                                                            |
|  5 | [Climatology of total parametrized zonally averaged GWD at 70 hPa in January averaged over the period 1980-2010](plots/tendency_comparison_new.pdf)                                                                                      | [reproduce_Fig5+S3.ipynb](code/reproduce_Fig5+S3.ipynb)                           |                                                                                                                        |
|  6 | [OGWD distribution at 70 hPa during boreal winter within HI, EA and WA hotspot, respectively, with log-norm fit](plots/OGWD_distribution_hotspots_DJFonly_wfits_lognorm.pdf)                                               | [reproduce_Fig6+S15.ipynb](code/reproduce_Fig6+S15.ipynb)                     | [averaging.py](code/averaging.py)                                                                                                                            |
|  7 | [Spatial and annual variability of the Gini index of GW momentum fluxes at 100, 70, 50, 20, 10, 1 hPa within HI, EA and WA hotspot, respectively](plots/gini_index_hotspots.pdf)                                                  | [reproduce_Fig7+Smaps+S12-S14.ipynb](code/reproduce_Fig7+Smaps+S12-S14.ipynb)               |                                                                                                                           |
|  8 | [Composite anomalies of OGWD averaged at all lags within the selected hotspot areas on the 20-day timescale](plots/accelogw_anomalies_all_20days_profiles_alllags_wsignificance_with-zonalwind.pdf)                                                  | [reproduce_Fig8.ipynb](code/reproduce_Fig8.ipynb)               | [anomalies_calc-accelogw.ipynb](code/anomalies_calc-accelogw.ipynb)                                                                                                                            |

#### Supplementary figures
|  #  | Figure                                                                                                                                                                                                    | Notebook                                                                              | Dependencies                                                                                                                                                             |
|:---:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|  S1 | [Comparison between zonal and meridional OGWD didstributions at 70 hPa during boreal winter within HI, EA and WA hotspot, respectively](plots/merOGWD_distribution_hotspots_DJFonly.pdf)                                               | [reproduce_FigS1.ipynb](code/reproduce_FigS1.ipynb)                     | |
| S2 | [Climatological relative EPFD contribution to the overall drag in January averaged over the period 1980-2010](plots/ratio_epfd_vs_total_january.pdf)                                                                              | [reproduce_FigS2.ipynb](code/reproduce_FigS2.ipynb)                       |                                                                                                                                    |
|  S3 | [Climatological average of particular CMAM-sd tendencies in January averaged over the period 1980-2010](plots/CMAM_january_climatological_comparison_origGWD_vs_GWDfromTEM_new.pdf)                                                                              | [reproduce_Fig5+S3.ipynb](code/reproduce_Fig5+S3.ipynb)                       |                                                                                                                                    |
|  S4 | [Spatial variability of Gini index of OGW momentum fluxes at 100,70,50,20,10,1 hPa for DJF](plots/gini_index_map_oro_months12-1-2.pdf)                                                                              | [reproduce_Fig7+Smaps+S12-S14.ipynb](code/reproduce_Fig7+Smaps+S12-S14.ipynb)                       |                                                                                                                                     |
|  S5 | [Spatial variability of Gini index of NOGW momentum fluxes at 100,70,50,20,10,1 hPa for DJF](plots/gini_index_map_noro_months12-1-2.pdf)                                                                              | [reproduce_Fig7+Smaps+S12-S14.ipynb](code/reproduce_Fig7+Smaps+S12-S14.ipynb)                       |                                                                                                                                    |
|  S6 | [Spatial variability of Gini index of OGW momentum fluxes at 100,70,50,20,10,1 hPa for MAM](plots/gini_index_map_oro_months3-4-5.pdf)                                                                              | [reproduce_Fig7+Smaps+S12-S14.ipynb](code/reproduce_Fig7+Smaps+S12-S14.ipynb)                       |                                                                                                                                    |
|  S7 | [Spatial variability of Gini index of NOGW momentum fluxes at 100,70,50,20,10,1 hPa for MAM](plots/gini_index_map_noro_months3-4-5.pdf)                                                                              | [reproduce_Fig7+Smaps+S12-S14.ipynb](code/reproduce_Fig7+Smaps+S12-S14.ipynb)                       |                                                                                                                                      |
|  S8 | [Spatial variability of Gini index of OGW momentum fluxes at 100,70,50,20,10,1 hPa for JJA](plots/gini_index_map_oro_months6-7-8.pdf)                                                                              | [reproduce_Fig7+Smaps+S12-S14.ipynb](code/reproduce_Fig7+Smaps+S12-S14.ipynb)                       |                                                                                                                                      |
|  S9 | [Spatial variability of Gini index of NOGW momentum fluxes at 100,70,50,20,10,1 hPa for JJA](plots/gini_index_map_noro_months6-7-8.pdf)                                                                              | [reproduce_Fig7+Smaps+S12-S14.ipynb](code/reproduce_Fig7+Smaps+S12-S14.ipynb)                       |                                                                                                                                      |
|  S10 | [Spatial variability of Gini index of OGW momentum fluxes at 100,70,50,20,10,1 hPa for SON](plots/gini_index_map_oro_months9-10-11.pdf)                                                                              | [reproduce_Fig7+Smaps+S12-S14.ipynb](code/reproduce_Fig7+Smaps+S12-S14.ipynb)                       |                                                                                                                                     |
|  S11 | [Spatial variability of Gini index of NOGW momentum fluxes at 100,70,50,20,10,1 hPa for SON](plots/gini_index_map_noro_months9-10-11.pdf)                                                                              | [reproduce_Fig7+Smaps+S12-S14.ipynb](code/reproduce_Fig7+Smaps+S12-S14.ipynb)                       |                                                                                                                                  |
|  S12 | [Spatial and annual variability of the Gini index of OGW momentum fluxes at 100, 70, 50, 20, 10, 1 hPa within HI, EA and WA hotspot, respectively](plots/gini_index_hotspots_erf-oro.pdf)                                                                              | [reproduce_Fig7+Smaps+S12-S14.ipynb]](code/reproduce_Fig7+Smaps+S12-S14.ipynb)                       |                                                                                                                                  |
|  S13 | [Spatial and annual variability of the Gini index of NOGW momentum fluxes at 100, 70, 50, 20, 10, 1 hPa within HI, EA and WA hotspot, respectively](plots/gini_index_hotspots_erf-noro.pdf)                                                                              | [reproduce_Fig7+Smaps+S12-S14.ipynb](code/reproduce_Fig7+Smaps+S12-S14.ipynb)                       |                                                                                                                                  |
|  S14 | [Spatial and annual variability of Gini index of OGW momentum fluxes at 850 hPa within Himalayas, East Asia and West America hotspot, respectively.](plots/gini_index_oro@850hPa.pdf)                                                                              | [reproduce_Fig7+Smaps+S12-S14.ipynb](code/reproduce_Fig7+Smaps+S12-S14.ipynb)                       |                                                                                                                                       |
|  S15 | [OGWD distribution at 70 hPa during boreal winter within HI, EA and WA hotspot, respectively with Weibull fit](plots/OGWD_distribution_hotspots_DJFonly_wfits_weibull_min.pdf)                                                                              | [reproduce_Fig6+S15.ipynb](code/reproduce_Fig6+S15.ipynb)                       |                                                                                                                                  |


### Tables
| #  | Figure                                                                                                                                | Notebook                                                    | Dependencies                                  |
|---:|:--------------------------------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------|:----------------------------------------------|
| 1 | [Number of detected peak events per month for the three selected hotspot areas](tables/ogwd_events.tex) | [reproduce_Fig3+table.ipynb](code/reproduce_Fig3+table.ipynb)            |  |

### Required package installation
`pip install -r requirements.txt`

### References
Kuchar, A., Sacha, P., Eichinger, R., Jacobi, C., Pisoft, P., and Rieder, H. E.: On the intermittency of orographic gravity wave hotspots and its importance for middle atmosphere dynamics, Weather Clim. Dynam. Discuss., [https://doi.org/10.5194/wcd-2020-21](https://doi.org/10.5194/wcd-2020-21), in review, 2020. 
