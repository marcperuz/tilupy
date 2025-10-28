# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 14:47:17 2022

@author: peruzzetto
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

import shapely.geometry as geom
import shapely.ops

import tilupy.read


def CSI(pred, obs):
    ipred = pred > 0
    iobs = obs > 0

    TP = np.sum(ipred * iobs)
    FP = np.sum(ipred * ~iobs)
    FN = np.sum(~ipred * iobs)

    return TP / (TP + FP + FN)


def diff_runout(
    x_contour, y_contour, point_ref, section=None, orientation="W-E"
):
    npts = len(x_contour)
    contour = geom.LineString(
        [(x_contour[i], y_contour[i]) for i in range(npts)]
    )
    point = geom.Point(point_ref)
    if section is None:
        return point.distance(contour)
    elif isinstance(section, np.ndarray):
        section = geom.LineString(section)

    assert isinstance(section, geom.LineString)
    section = revert_line(section, orientation)
    intersections = section.intersection(contour)
    if isinstance(intersections, geom.MultiPoint):
        intersections = geom.LineString(intersections.geoms)
    intersections = np.array(intersections.coords)
    if orientation == "W-E":
        i = np.argmax(intersections[:, 0])
    if orientation == "E-W":
        i = np.argmin(intersections[:, 0])
    if orientation == "S-N":
        i = np.argmax(intersections[:, 1])
    if orientation == "N-S":
        i = np.argmin(intersections[:, 1])
    intersection = geom.Point(intersections[i, :])

    #######
    # plt.figure()
    # cont = np.array(contour.coords)
    # sec = np.array(section.coords)
    # inter = np.array(intersection.coords)
    # pt = np.array(point.coords)
    # plt.plot(cont[:,0], cont[:,1],
    #           sec[:,0], sec[:,1],
    #           pt[:,0], pt[:,1], 'o',
    #           inter[:,0], inter[:,1],'x')
    #######

    return section.project(intersection) - section.project(point)


def revert_line(line, orientation="W-E"):
    pt_init = line.coords[0]
    pt_end = line.coords[-1]
    if orientation == "W-E":
        if pt_init[0] > pt_end[0]:
            line = shapely.ops.substring(line, 1, 0, normalized=True)
    elif orientation == "E-W":
        if pt_init[0] < pt_end[0]:
            line = shapely.ops.substring(line, 1, 0, normalized=True)
    if orientation == "S-N":
        if pt_init[1] > pt_end[1]:
            line = shapely.ops.substring(line, 1, 0, normalized=True)
    elif orientation == "N-S":
        if pt_init[1] < pt_end[1]:
            line = shapely.ops.substring(line, 1, 0, normalized=True)

    return line


def get_contour(x, y, z, zlevels, indstep=1, maxdist=30, closed_contour=True):
    # Add sup_value at the border of the array, to make sure contour
    # lines are closed
    if closed_contour:
        z2 = z.copy()
        ni = z2.shape[0]
        nj = z2.shape[1]
        z2 = np.vstack([np.zeros((1, nj)), z2, np.zeros((1, nj))])
        z2 = np.hstack([np.zeros((ni + 2, 1)), z2, np.zeros((ni + 2, 1))])
        dxi = x[1] - x[0]
        dxf = x[-1] - x[-2]
        dyi = y[1] - y[0]
        dyf = y[-1] - y[-2]
        x2 = np.insert(np.append(x, x[-1] + dxf), 0, x[0] - dxi)
        y2 = np.insert(np.append(y, y[-1] + dyf), 0, y[0] - dyi)
    else:
        x2, y2, z2 = x, y, z

    backend = plt.get_backend()
    plt.switch_backend("Agg")
    fig = plt.figure()
    ax = plt.gca()
    cs = ax.contour(x2, y2, np.flip(z2, 0), zlevels)
    nn1 = 1
    v1 = np.zeros((1, 2))
    xcontour = {}
    ycontour = {}
    for indlevel in range(len(zlevels)):
        levels = cs.allsegs[indlevel]
        for p in levels:
            if p.shape[0] > nn1:
                v1 = p
                nn1 = p.shape[0]
        xc = [v1[::indstep, 0]]
        yc = [v1[::indstep, 1]]
        if maxdist is not None and not closed_contour:
            ddx = np.abs(v1[0, 0] - v1[-1, 0])
            ddy = np.abs(v1[0, 1] - v1[-1, 1])
            dd = np.sqrt(ddx**2 + ddy**2)
            if dd > maxdist:
                xc[0] = None
                yc[0] = None
        xcontour[zlevels[indlevel]] = xc[0]
        ycontour[zlevels[indlevel]] = yc[0]
    plt.close(fig)
    plt.switch_backend(backend)
    return xcontour, ycontour


def get_profile(data: tilupy.read.TemporalResults2D | tilupy.read.StaticResults2D,
                extraction_mode: str = "axis",
                data_threshold: float = 1e-3,
                **extraction_params,
                ) -> tuple[tilupy.read.TemporalResults1D, float | np.ndarray]:
    """Extract profile with different modes and options.

    Parameters
    ----------
    data : tilupy.read.TemporalResults2D or tilupy.read.StaticResults2D
        Data to extract the profile from.
    extraction_mode : str, optional
        Method to extract profiles:
        
            - "axis": Extracts a profile along an axis.
            - "coordinates": Extracts a profile along specified coordinates.
            - "shapefile": Extracts a profile along a shapefile (polylines).
        
        Be default "axis".
        
    data_threshold : float, optional
        Minimum value to consider as part of the profile, by default 1e-3.
        
    extraction_params : dict, optional
        Different parameters to be entered depending on the extraction method chosen:
        
            - If :data:`extraction_mode == "axis"`:
            
                - :data:`axis`: str, optional
                    Axis where to extract the profile ['X', 'Y'], by default 'Y'.
                - :data:`profile_position`: float, optional
                    Position where to extract the profile. If None choose the median.
                    By default None.
                Must be read: profile in `axis` = `profile_position m`.
            
            - If :data:`extraction_mode == "coordinates"`:
            
                - :data:`xcoord`: numpy.ndarray, optional
                    X coordinates of the profile, by default :attr:`_x`.
                - :data:`ycoord`: numpy.ndarray, optional
                    Y coordinates of the profile, by default :data:`[0., 0., ...]`.
            
            - If :data:`extraction_mode == "shapefile"`:
            
                - :data:`path`: str
                    Path to the shapefile.
                - :data:`x_origin`: float, optional
                    Value of the X coordinate of the origin (top-left corner) in the shapefile's coordinate system, by default 0.0 (EPSG:2154).
                - :data:`y_origin`: float, optional
                    Value of the y coordinate of the origin (top-left corner) in the shapefile's coordinate system, by default :data:`_y[-1]` (EPSG:2154).
                - :data:`x_pixsize`: float, optional
                    Size of a pixel along the X coordinate in the shapefile's coordinate system, by default :data:`_x[1] - _x[0]` (EPSG:2154).
                - :data:`y_pixsize`: float, optional
                    Size of a pixel along the Y coordinate in the shapefile's coordinate system, by default :data:`_y[1] - _y[0]` (EPSG:2154).
                - :data:`step`: float, optional
                    Spatial step between profile points, by default 10.0.
                    
        By default None
        
    Returns
    -------
    tilupy.read.TemporalResults1D
        Extracted profiles.
    float or numpy.ndarray
        Specific output depending on :data:`extraction_mode`:
        
            - If :data:`extraction_mode == "axis"`: float
                Position of the profile.
            - If :data:`extraction_mode == "coordinates"`: tuple[numpy.ndarray]
                X coordinates, Y coordinates and distance values.
            - If :data:`extraction_mode == "shapefile"`: numpy.ndarray
                Distance values.
                            
    Raises
    ------
    ValueError
        If :data:`extraction_mode == "axis"` and if invalid :data:`axis`.
    ValueError
        If :data:`extraction_mode == "axis"` and if invalid format for :data:`profile_position`.
    ValueError
        If :data:`extraction_mode == "axis"` and if no value position found in axis.
    ValueError
        If :data:`extraction_mode == "coordinates"` and if invalid format for :data:`xcoord` or :data:`ycoord`.
    ValueError
        If :data:`extraction_mode == "coordinates"` and if invalid dimension for :data:`xcoord` or :data:`ycoord`.
    ValueError
        If :data:`extraction_mode == "coordinates"` and if :data:`xcoord` and :data:`ycoord` doesn't have same size.
    ValueError
        If :data:`extraction_mode == "shapefile"` and if no :data`path` is given.
    ValueError
        If :data:`extraction_mode == "shapefile"` and if invalid format for :data:`x_origin`, :data:`y_origin`, :data:`x_pixsize`, :data:`y_pixsize` or :data:`step`.
    TypeError
        If :data:`extraction_mode == "shapefile"` and if invalid geometry for the shapefile.
    ValueError
        If :data:`extraction_mode == "shapefile"` and if no linestring found in the shapefile.
    ValueError
        If invalid :data:`extraction_mode`.
    """
    if not isinstance(data, tilupy.read.TemporalResults2D) and not isinstance(data, tilupy.read.StaticResults2D):
        raise ValueError("Can only extract profile from 2D data.")
    
    y_coord, x_coord = data.y, data.x
    y_size, x_size, = len(y_coord), len(x_coord)
    data_field = data.d.copy()
    
    # Apply mask on data
    data_field[data_field <= data_threshold] = 0

    extraction_params = {} if extraction_params is None else extraction_params

    if extraction_mode == "axis":
        # Create specific params if not given
        if "axis" not in extraction_params:
            extraction_params["axis"] = 'Y'
        if "profile_position" not in extraction_params:
            extraction_params["profile_position"] = None
        
        # Check errors
        if extraction_params["axis"] not in ['x', 'X', 'y', 'Y']:
            raise ValueError("Invalid axis: 'X' or 'Y'.")
        extraction_params["axis"] = extraction_params["axis"].upper()
        
        # Depending on "profile_position" type, choose median or position value
        if extraction_params["profile_position"] is None:
            if extraction_params["axis"] == 'X':
                extraction_params["profile_index"] = x_size//2
                closest_value=x_coord[extraction_params["profile_index"]]
            else:
                extraction_params["profile_index"] = y_size//2
                closest_value=y_coord[extraction_params["profile_index"]]
        
        elif isinstance(extraction_params["profile_position"], float) or isinstance(extraction_params["profile_position"], int):
            coord_val = extraction_params["profile_position"]
            x_index, y_index = None, None
            
            if extraction_params["axis"] == 'X':
                if not isinstance(x_coord, np.ndarray):
                    x_coord = np.array(x_coord)
                
                x_index = np.argmin(np.abs(x_coord - coord_val))
                closest_value = x_coord[x_index]
                
                if x_index is None:
                    raise ValueError(f"Find no values, must be: {x_coord}")
                extraction_params["profile_index"] = x_index
            else:
                if not isinstance(y_coord, np.ndarray):
                    y_coord = np.array(y_coord)
                
                y_index = np.argmin(np.abs(y_coord - coord_val))
                closest_value = y_coord[y_index]
                
                if y_index is None:
                    raise ValueError(f"Find no values, must be: {y_coord}")
                extraction_params["profile_index"] = y_index
                            
        else:
            raise ValueError("Invalid format for 'profile_position'. Must be None or float position.")

        # Return profiles
        if extraction_params["axis"] == 'X':
            if isinstance(data, tilupy.read.TemporalResults2D):
                return (tilupy.read.TemporalResults1D(name=data.name,
                                                      d=data_field[:, extraction_params["profile_index"], :],
                                                      t=data.t,
                                                      coords=data.y,
                                                      coords_name='y',
                                                      notation=data.notation),
                        closest_value)
            else:
                return (tilupy.read.StaticResults1D(name=data.name,
                                                    d=data_field[:, extraction_params["profile_index"]],
                                                    coords=data.y,
                                                    coords_name='y',
                                                    notation=data.notation),
                        closest_value)
        else:
            if isinstance(data, tilupy.read.TemporalResults2D):
                return (tilupy.read.TemporalResults1D(name=data.name,
                                                      d=data_field[extraction_params["profile_index"], :, :],
                                                      t=data.t,
                                                      coords=data.x,
                                                      coords_name='x',
                                                      notation=data.notation),
                        closest_value)
            else:
                return (tilupy.read.StaticResults1D(name=data.name,
                                                    d=data_field[extraction_params["profile_index"], :],
                                                    coords=data.x,
                                                    coords_name='x',
                                                    notation=data.notation),
                        closest_value)
        
    elif extraction_mode == "coordinates":
        if "xcoord" not in extraction_params:
            extraction_params["xcoord"] = x_coord[:]
        if "ycoord" not in extraction_params:
            extraction_params["ycoord"] = [0] * len(x_coord)
        
        # Check errors
        if not isinstance(extraction_params["xcoord"], np.ndarray):
            if isinstance(extraction_params["xcoord"], list):
                extraction_params["xcoord"] = np.array(extraction_params["xcoord"])
            else:
                raise ValueError("Invalid format for 'xcoord'. Must be a numpy array.")
        if not isinstance(extraction_params["ycoord"], np.ndarray):
            if isinstance(extraction_params["ycoord"], list):
                extraction_params["ycoord"] = np.array(extraction_params["ycoord"])
            else:
                raise ValueError("Invalid format for 'ycoord'. Must be a numpy array.")
        
        if extraction_params["xcoord"].ndim != 1:
            raise ValueError("Invild dimension. 'xcoord' must be a 1d array.")
        if extraction_params["ycoord"].ndim != 1:
            raise ValueError("Invild dimension. 'ycoord' must be a 1d array.")
        
        if len(extraction_params["xcoord"]) != len(extraction_params["ycoord"]):
            raise ValueError(f"'xcoord' and 'ycoord' must have same size: ({len(extraction_params["xcoord"])}, {len(extraction_params["ycoord"])})")
        
        # Extract index from nearest value of xcoord and ycoord
        x_distances = np.abs(x_coord[None, :] - extraction_params["xcoord"][:, None])
        y_distances = np.abs(y_coord[None, :] - extraction_params["ycoord"][:, None])
        
        x_indexes = np.argmin(x_distances, axis=1)
        y_indexes = np.argmin(y_distances, axis=1)

        # Compute distance
        x_values = x_coord[x_indexes]
        y_values = y_coord[y_indexes]
        
        dx = np.diff(x_values)
        dy = np.diff(y_values)
        
        distance = np.sqrt(dx**2 + dy**2)
        distance = np.concatenate(([0], np.cumsum(distance)))

        # Return profile
        if isinstance(data, tilupy.read.TemporalResults2D):
            return (tilupy.read.TemporalResults1D(name=data.name,
                                                  d=data_field[y_indexes, x_indexes, :],
                                                  t=data.t,
                                                  coords=distance,
                                                  coords_name='dist'),
                    (x_values, 
                     y_values, 
                     distance))
        else:
            return (tilupy.read.StaticResults1D(name=data.name,
                                                d=data_field[y_indexes, x_indexes],
                                                coords=distance,
                                                coords_name='dist'),
                    (x_values, 
                     y_values, 
                     distance))

    elif extraction_mode == "shapefile" :
        if "path" not in extraction_params:
            extraction_params["path"] = None
        if "x_origin" not in extraction_params:
            extraction_params["x_origin"] = 0
        if "y_origin" not in extraction_params:
            extraction_params["y_origin"] = y_coord[-1]
        if "x_pixsize" not in extraction_params:
            extraction_params["x_pixsize"] = x_coord[1] - x_coord[0]
        if "y_pixsize" not in extraction_params:
            extraction_params["y_pixsize"] = y_coord[1] - y_coord[0]
        if "step" not in extraction_params:
            extraction_params["step"] = 10
        
        # Check errors
        if extraction_params["path"] is None:
            raise ValueError("No path to the shape file given.")

        if not isinstance(extraction_params["x_origin"], float) and not isinstance(extraction_params["x_origin"], int):
            raise ValueError("'x_origin' must be float.")
        if not isinstance(extraction_params["y_origin"], float) and not isinstance(extraction_params["y_origin"], int):
            raise ValueError("'y_origin' must be float.")
        if not isinstance(extraction_params["x_pixsize"], float) and not isinstance(extraction_params["x_pixsize"], int):
            raise ValueError("'x_pixsize' must be float.")
        if not isinstance(extraction_params["y_pixsize"], float) and not isinstance(extraction_params["y_pixsize"], int):
            raise ValueError("'y_pixsize' must be float.")
        
        if not isinstance(extraction_params["step"], float) and not isinstance(extraction_params["step"], int):
            raise ValueError("'step' must be float.")
        
        # Import specific module and define extraction function
        from shapely.geometry import LineString, MultiLineString
        from shapely.ops import linemerge
        import geopandas as gpd
        from affine import Affine
        
        def extract_lines_from_shp_file(shapefile_path):
            """Extract LineString objects from a shapefile.
            """
            gdf = gpd.read_file(shapefile_path)
            lines = []
            for geom in gdf.geometry:
                if isinstance(geom, LineString):
                    lines.append(geom)
                elif isinstance(geom, MultiLineString):
                    lines.extend(list(geom))
                else:
                    raise TypeError(f"Invalid geometry: {type(geom)}")
            if not lines:
                raise ValueError("No Linestring found.")
            return lines
        
        lines = extract_lines_from_shp_file(extraction_params["path"])
        
        # If multiple lines, merge them together
        merged = linemerge(lines)
        if isinstance(merged, LineString):
            profile_line = merged
        else:
            profile_line = LineString([pt for line in merged for pt in line.coords])
        
        # Construct the affine transformation
        transform = (Affine.translation(extraction_params["x_origin"], 
                                        extraction_params["y_origin"]) 
                        * Affine.scale(extraction_params["x_pixsize"], 
                                    -extraction_params["y_pixsize"]))
        inv = ~transform  # invert : (x, y) -> (row, col)

        distances = np.arange(0, 
                                profile_line.length, 
                                extraction_params["step"])
        points = [profile_line.interpolate(d) for d in distances]

        # Conversion coordinates -> indexes
        rowcols = [inv * (pt.x, pt.y) for pt in points]
        rowcols = [(int(round(r)), int(round(c))) for c, r in rowcols]

        # Extract values
        all_values = []
        for t in range(len(data.t)):
            values = []
            valid_distances = []
            for d, (r, c) in zip(distances, rowcols):
                if 0 <= r < data_field.shape[0] and 0 <= c < data_field.shape[1]:
                    values.append(data_field[r, c, t])
                    valid_distances.append(d)
            all_values.append(values)
        
        all_values = np.array(all_values)
        valid_distances = np.array(valid_distances)

        # Return profile
        if isinstance(data, tilupy.read.TemporalResults2D):
            return (tilupy.read.TemporalResults1D(name=data.name,
                                                  d=all_values.T,
                                                  t=data.t,
                                                  coords=valid_distances,
                                                  coords_name="dist",
                                                  notation=data.notation),
                    valid_distances)
        else:
            return (tilupy.read.StaticResults1D(name=data.name,
                                                d=all_values.T,
                                                coords=valid_distances,
                                                coords_name="dist",
                                                notation=data.notation),
                    valid_distances)
    else :
        raise ValueError("Invalid 'extraction_mode': 'axis', 'coordinates' or 'shapefile'.")


def format_path_linux(path):
    """
    Change a Windows-type path to a path formatted for Linux. \\ are changed
    to /, and partitions like "C:" are changed to "/mnt/c/"

    Parameters
    ----------
    path : string
        String with the path to be modified.

    Returns
    -------
    path2 : string
        Formatted path.

    """
    if path[1] == ":":
        path2 = "/mnt/{:s}/".format(path[0].lower()) + path[2:]
    else:
        path2 = path
    path2 = path2.replace("\\", "/")
    if " " in path2:
        path2 = '"' + path2 + '"'
    return path2
