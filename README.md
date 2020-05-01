# On the intermittency of orographic gravity wave hotspots and its importance for middle atmosphere dynamics
**A. Kuchar, P. Sacha, R. Eichinger, Ch. Jacobi, P. Pisoft, and H. Rieder**

Code used to process and visualise the model and other data outputs in order to reproduce figures in the manuscript.
Model data are available [here](http://climate-modelling.canada.ca/climatemodeldata/cmam/output/CMAM/CMAM30-SD/index.shtml).

Notebooks for each individual figure as well as for two data tables are in the [`code/` directory](code), while the figures themselves are in the `plots/` directory.

### Figures
|  #  | Figure                                                                                                                                                                                                    | Notebook                                                                              | Dependencies                                                                                                                                                             |
|:---:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|  1 | [Ratio of zonally averaged OGWD in zonal direction to the total wave forcing](plots/ratio_oGWD_vs_PWD+noGWD_climatology_wSSW_all.pdf)                                                                              | [reproduce_Fig1+2.ipynb](code/reproduce_Fig1+2.ipynb)                       |                                                                                                                                       |
|  2 | [Winter climatology of the zonal OGWD component at 70 hPa](plots/oGWD_map@70hPa_climatology_hotspots_DJF.pdf)                                                      | [reproduce_Fig1+2.ipynb](code/reproduce_Fig1+2.ipynb)                 |                                                                                                                           |
|  3 | [Time average vertical profiles of temperature and water vapor at the sub-stellar point and its antipode](plots/static_ts_plot_ogwd_hotspots@70hPa.pdf)                | [reproduce_Fig3.ipynb](code/reproduce_Fig3.ipynb)                 | [averaging.py](code/averaging.py), [detect_peaks.py](code/detect_peaks.py)                                                                                                                            |
|  4 | [Climatologies for the absolute gravity wave momentum fluxes at 20 and 30 km](plots/CMAM_vs_GRACILE_DJF.pdf) | [reproduce_Fig4.ipynb](code/reproduce_Fig4.ipynb)                     |                                                                                                                            |
|  5 | [Climatology of total parametrized zonally averaged GWD at 70 hPa in January averaged over the period 1980-2010](plots/tendency_comparison_new.pdf)                                                                                      | [reproduce_Fig5+S2.ipynb](code/reproduce_Fig5+S2.ipynb)                           |                                                                                                                        |
|  6 | [OGWD distribution at 70 hPa during boreal winter within HI, EA and WA hotspot, respectively](plots/OGWD_distribution_hotspots_DJFonly.pdf)                                               | [reproduce_Fig6.ipynb](code/reproduce_Fig6.ipynb)                     | [averaging.py](code/averaging.py)                                                                                                                            |
|  7 | [Spatial and annual variability of the Gini index of GW momentum fluxes at 100, 70, 50, 20, 10, 1 hPa within HI, EA and WA hotspot, respectively](plots/gini_index_hotspots.pdf)                                                  | [reproduce_Fig7+S3+Smaps+S12+S13.ipynb](code/reproduce_Fig7+S3+Smaps+S12+S13.ipynb)               |                                                                                                                           |
|  8 | [Composite anomalies of OGWD averaged at all lags within the selected hotspot areas on the 20-day timescale](plots/accelogw_anomalies_all_20days_profiles_alllags_wsignificance_with-zonalwind.pdf)                                                  | [reproduce_Fig8.ipynb](code/reproduce_Fig8.ipynb)               | [anomalies_calc-accelogw.ipynb](code/anomalies_calc-accelogw.ipynb)                                                                                                                            |

#### Supplementary figures
|  #  | Figure                                                                                                                                                                                                    | Notebook                                                                              | Dependencies                                                                                                                                                             |
|:---:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|  B1 | [Climatological relative EPFD contribution to the overall drag in January averaged over the period 1980-2010](plots/ratio_epfd_vs_total_january.pdf)                                                                              | [reproduce_FigS1.ipynb](code/reproduce_FigS1.ipynb)                       |                                                                                                                                    |
|  B2 | [Climatological average of particular CMAM-sd tendencies in January averaged over the period 1980-2010](plots/CMAM_january_climatological_comparison_origGWD_vs_GWDfromTEM_new.pdf)                                                                              | [reproduce_Fig1+2.ipynb](code/reproduce_Fig5+S2.ipynb)                       |                                                                                                                                    |
|  B3 | [Spatial and annual variability of Gini index of OGW momentum fluxes at 850 hPa within Himalayas, East Asia and West America hotspot, respectively.](plots/gini_index_oro@850hPa.pdf)                                                                              | [reproduce_Fig7+S3+Smaps+S12+S13.ipynb](code/reproduce_Fig7+S3+Smaps+S12+S13.ipynb)                       |                                                                                                                                       |
|  B4 | [Spatial variability of Gini index of OGW momentum fluxes at 100,70,50,20,10,1 hPa for DJF](plots/gini_index_map_oro_months12-1-2.pdf)                                                                              | [reproduce_Fig7+S3+Smaps+S12+S13.ipynb](code/reproduce_Fig7+S3+Smaps+S12+S13.ipynb)                       |                                                                                                                                     |
|  B5 | [Spatial variability of Gini index of OGW momentum fluxes at 100,70,50,20,10,1 hPa for DJF](plots/gini_index_map_oro_months12-1-2.pdf)                                                                              | [reproduce_Fig7+S3+Smaps+S12+S13.ipynb](code/reproduce_Fig7+S3+Smaps+S12+S13.ipynb)                       |                                                                                                                                    |
|  B6 | [Spatial variability of Gini index of OGW momentum fluxes at 100,70,50,20,10,1 hPa for MAM](plots/gini_index_map_oro_months3-4-5.pdf)                                                                              | [reproduce_Fig7+S3+Smaps+S12+S13.ipynb](code/reproduce_Fig7+S3+Smaps+S12+S13.ipynb)                       |                                                                                                                                    |
|  B7 | [Spatial variability of Gini index of NOGW momentum fluxes at 100,70,50,20,10,1 hPa for MAM](plots/gini_index_map_noro_months3-4-5.pdf)                                                                              | [reproduce_Fig7+S3+Smaps+S12+S13.ipynb](code/reproduce_Fig7+S3+Smaps+S12+S13.ipynb)                       |                                                                                                                                      |
|  B8 | [Spatial variability of Gini index of OGW momentum fluxes at 100,70,50,20,10,1 hPa for JJA](plots/gini_index_map_oro_months6-7-8.pdf)                                                                              | [reproduce_Fig7+S3+Smaps+S12+S13.ipynb](code/reproduce_Fig7+S3+Smaps+S12+S13.ipynb)                       |                                                                                                                                      |
|  B9 | [Spatial variability of Gini index of NOGW momentum fluxes at 100,70,50,20,10,1 hPa for JJA](plots/gini_index_map_noro_months6-7-8.pdf)                                                                              | [reproduce_Fig7+S3+Smaps+S12+S13.ipynb](code/reproduce_Fig7+S3+Smaps+S12+S13.ipynb)                       |                                                                                                                                      |
|  B10 | [Spatial variability of Gini index of OGW momentum fluxes at 100,70,50,20,10,1 hPa for DJF](plots/gini_index_map_oro_months9-10-11.pdf)                                                                              | [reproduce_Fig7+S3+Smaps+S12+S13.ipynb](code/reproduce_Fig7+S3+Smaps+S12+S13.ipynb)                       |                                                                                                                                     |
|  B11 | [Spatial variability of Gini index of NOGW momentum fluxes at 100,70,50,20,10,1 hPa for SON](plots/gini_index_map_noro_months9-10-11.pdf)                                                                              | [reproduce_Fig7+S3+Smaps+S12+S13.ipynb](code/reproduce_Fig7+S3+Smaps+S12+S13.ipynb)                       |                                                                                                                                  |
|  B12 | [Spatial and annual variability of the Gini index of OGW momentum fluxes at 100, 70, 50, 20, 10, 1 hPa within HI, EA and WA hotspot, respectively](plots/gini_index_hotspots_erf-oro.pdf)                                                                              | [reproduce_Fig7+S3+Smaps+S12+S13.ipynb](code/reproduce_Fig7+S3+Smaps+S12+S13.ipynb)                       |                                                                                                                                  |
|  B13 | [Spatial and annual variability of the Gini index of NOGW momentum fluxes at 100, 70, 50, 20, 10, 1 hPa within HI, EA and WA hotspot, respectively](plots/gini_index_hotspots_erf-noro.pdf)                                                                              | [reproduce_Fig7+S3+Smaps+S12+S13.ipynb](code/reproduce_Fig7+S3+Smaps+S12+S13.ipynb)                       |                                                                                                                                  |


### Tables
| #  | Figure                                                                                                                                | Notebook                                                    | Dependencies                                  |
|---:|:--------------------------------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------|:----------------------------------------------|
| 1 | [Number of detected peak events per month for the three selected hotspot areas](tables/ogwd_events.tex) | [reproduce_Fig3.ipynb](code/reproduce_Fig3.ipynb)            |  |
