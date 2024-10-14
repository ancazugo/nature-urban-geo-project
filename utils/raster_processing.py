
import numpy as np
import xarray as xr

def resample_raster(da: xr.DataArray, target_resolution: int, method: str='downsample'):
    """
    Resamples an xarray DataArray (raster) to a different resolution.
    
    Parameters:
    da (xarray.DataArray): The input raster.
    target_resolution (tuple): A tuple of the form (new_lon_res, new_lat_res) for the new resolution.
    method (str): Either 'downsample' or 'upsample'. 'downsample' will reduce resolution by averaging,
                  and 'upsample' will increase resolution by interpolation.
                  
    Returns:
    xarray.DataArray: The resampled raster.
    """
    
    # Get current resolution
    lon_res = da.lon[1] - da.lon[0]
    lat_res = da.lat[1] - da.lat[0]

    # Target resolution
    target_lon_res, target_lat_res = target_resolution
    
    if method == 'downsample':
        # Calculate factors for coarsening (downsampling)
        lon_factor = int(target_lon_res / lon_res)
        lat_factor = int(target_lat_res / lat_res)

        if lon_factor < 1 or lat_factor < 1:
            raise ValueError("Target resolution must be larger than current resolution for downsampling.")
        
        # Downsample by coarsening (mean aggregation)
        da_resampled = da.coarsen(lat=lat_factor, lon=lon_factor, boundary="trim").mean()

    elif method == 'upsample':
        # Create new coordinates for upsampling (interpolation)
        new_lons = xr.DataArray(
            np.arange(da.lon.min(), da.lon.max(), target_lon_res), dims="lon"
        )
        new_lats = xr.DataArray(
            np.arange(da.lat.min(), da.lat.max(), target_lat_res), dims="lat"
        )
        
        # Upsample using interpolation
        da_resampled = da.interp(lon=new_lons, lat=new_lats)

    else:
        raise ValueError("Method must be 'downsample' or 'upsample'.")

    return da_resampled