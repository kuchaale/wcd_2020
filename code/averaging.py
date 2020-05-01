# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"></ul></div>

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"></ul></div>

'''
Weighted averaging for xarray data sets, based on:
https://github.com/pydata/xarray/issues/422#issuecomment-140823232

Authors
-------
Mathias Hauser (https://github.com/mathause)
Xylar Asay-Davis
'''

import xarray


def xarray_average(data, dim=None, weights=None, **kwargs):  # {{{
    """
    weighted average for xarray objects

    Parameters
    ----------
    data : Dataset or DataArray
        the xarray object to average over

    dim : str or sequence of str, optional
        Dimension(s) over which to apply average.

    weights : DataArray
        weights to apply. Shape must be broadcastable to shape of data.

    kwargs : dict
        Additional keyword arguments passed on to internal calls to ``mean``
        or ``sum`` (performed on the data set or data array but *not* those
        performed on the weights)

    Returns
    -------
    reduced : Dataset or DataArray
        New xarray object with average applied to its data and the indicated
        dimension(s) removed.

    Authors
    -------
    Mathias Hauser (https://github.com/mathause)
    Xylar Asay-Davis
    """

    if isinstance(data, xarray.Dataset):
        return _average_ds(data, dim, weights, **kwargs)
    elif isinstance(data, xarray.DataArray):
        return _average_da(data, dim, weights, **kwargs)
    else:
        raise ValueError("date must be an xarray Dataset or DataArray")
    # }}}


def _average_da(da, dim=None, weights=None, **kwargs):  # {{{
    """
    weighted average for DataArrays

    Parameters
    ----------
    dim : str or sequence of str, optional
        Dimension(s) over which to apply average.

    weights : DataArray
        weights to apply. Shape must be broadcastable to shape of self.

    kwargs : dict
        Additional keyword arguments passed on to internal calls to ``mean``
        or ``sum`` (performed on the data set or data array but *not* those
        performed on the weights)

    Returns
    -------
    reduced : DataArray
        New DataArray with average applied to its data and the indicated
        dimension(s) removed.

    Authors
    -------
    Mathias Hauser (https://github.com/mathause)
    Xylar Asay-Davis
    """

    if weights is None:
        return da.mean(dim, **kwargs)
    else:
        if not isinstance(weights, xarray.DataArray):
            raise ValueError("weights must be a DataArray")

        # if NaNs are present, we need individual weights
        if da.notnull().any():
            total_weights = weights.where(da.notnull()).sum(dim=dim)
        else:
            total_weights = weights.sum(dim)

        return (da * weights).sum(dim, **kwargs) / total_weights  # }}}


def _average_ds(ds, dim=None, weights=None, **kwargs):  # {{{
    """
    weighted average for Datasets

    Parameters
    ----------
    dim : str or sequence of str, optional
        Dimension(s) over which to apply average.

    weights : DataArray
        weights to apply. Shape must be broadcastable to shape of data.

    kwargs : dict
        Additional keyword arguments passed on to internal calls to ``mean``
        or ``sum`` (performed on the data set or data array but *not* those
        performed on the weights)


    Returns
    -------
    reduced : Dataset
        New Dataset with average applied to its data and the indicated
        dimension(s) removed.

    Authors
    -------
    Mathias Hauser (https://github.com/mathause)
    Xylar Asay-Davis
    """

    if weights is None:
        return ds.mean(dim, **kwargs)
    else:
        return ds.apply(_average_da, dim=dim, weights=weights, **kwargs)  # }}}

# vim: foldmethod=marker ai ts=4 sts=4 et sw=4 ft=python
