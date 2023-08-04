import os
import re
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import datetime as dt
from scipy.optimize import nnls
from scipy.signal import lfilter

from lasio import LASFile, CurveItem

"""
Log contains parent classes to work with log data.

The Log class is subclassed from lasio LASFile class, which provide a
data structure. The methods are for petrophysical calculations and for
viewing data with the LogViewer class.

"""


class Log(LASFile):
    """
    Log

    Subclass of LASFile to provide an extension for all petrophysical
    calculations.

    Parameters
    ----------
    file_ref : str
        str path to las file
    drho_matrix : float (default 2.71)
        Matrix density for conversion from density porosity to density.
    kwargs : kwargs
        Key Word arguements for use with lasio LASFile class.

    Example
    -------
    >>> import petropy as ptr
    >>> # define path to las file
    >>> p = 'path/to/well.las'
    >>> # loads specified las file
    >>> log = ptr.Log(p)

    """
    
    def __init__(self, file_ref = None, drho_matrix = 2.71, **kwargs):

        if file_ref is not None:
            LASFile.__init__(self, file_ref = file_ref,
                             autodetect_encoding = True, **kwargs)

        # self.precondition(drho_matrix = drho_matrix)

        # self.fluid_properties_parameters_from_csv()
        # self.multimineral_parameters_from_csv()
        # self.tops = {}


    def precondition(self, drho_matrix = 2.71, n = 15):
        """
        Preconditions log curve by aliasing names.

        Precondition is used after initializing data and standardizes
        names for future calculations.

        Parameters
        ----------
        drho_matrix : float, optional
            drho_matrix is for converting density porosity to bulk
            densty, and is only used when bulk density is missing.
            Default value for limestone matrix. If log was run on
            sandstone matrix, use 2.65. If log was run on dolomite
            matrix, use 2.85.

        Note
        -----
            1. Curve Alias is provided by the curve_alias.xml file

        """

        file_dir = os.path.dirname(__file__)
        ALIAS_XML_PATH = os.path.join(file_dir, 'alias_folder', 'curve_alias.xml')

        if not os.path.isfile(ALIAS_XML_PATH):
            raise ValueError('Could not find alias xml at: %s' % ALIAS_XML_PATH)

        with open(ALIAS_XML_PATH, 'r') as f:
            root = ET.fromstring(f.read())

        # Create a dictionary to store the alias mappings
        alias_mappings = {}

        for log_alias_entry in root.findall(".//LogAliasEntry"):
            primary_log = log_alias_entry.find("PrimaryLog").text
            aliases = [alias.text for alias in log_alias_entry.findall("Alias")]
            alias_mappings[primary_log] = aliases

        # Change the names of the keys based on alias mappings and keep only the first primary log if there are multiple
        processed_keys = set()
        for primary_log, aliases in alias_mappings.items():
            for alias in aliases:
                if alias in self.keys() and alias not in processed_keys:
                    self[primary_log] = self[alias]
                    processed_keys.add(alias)
                    break

        if 'RHOB' not in self.keys() and 'DPHI' in self.keys():
                non_null_depth_mask = (self['DPHI'] != self.well.NULL.value)
                non_null_depths = self['DPHI'][non_null_depth_mask]
                calculated_rho = np.empty(len(self[0]))
                calculated_rho[non_null_depth_mask] = drho_matrix - (drho_matrix - 1) * non_null_depths
                self.append_curve('RHOB', calculated_rho, unit='g/cc',
                                value='',
                                descr='Calculated bulk density from density porosity assuming rho matrix = %.2f' % drho_matrix)

        # Filter the curves to keep only those that are in standard_curves list
        standard_curves = ['DEPT', 'GR', 'NPHI', 'RHOB', 'ILD', 'PE', 'DT']
        self_keys = list(self.keys())
        for curve in self_keys:
            if curve not in standard_curves:
                self.delete_curve(curve)

        # Check if any curves from standard_curves are present after filtering
        remaining_curves = [curve for curve in self.keys() if curve in standard_curves]
        if not remaining_curves:
            raise ValueError("None of the curves from the standard_curves list are present in the data.")

        def apply_lfilter(curve_data):
            valid_mask = ~np.logical_or(np.isnan(curve_data), curve_data == self.well.NULL.value)
            filtered_data = np.empty_like(curve_data)
            filtered_data[valid_mask] = lfilter(np.ones(n) / n, 1, curve_data[valid_mask])
            return np.where(valid_mask, filtered_data, np.nan)

        if 'DT' in self.keys():
            # Get DT curve data
            dt = self['DT']

            # Apply lfilter and add filtered curve to log
            dt_filtered = apply_lfilter(dt)
            self.append_curve('DT', dt_filtered, unit=self.curvesdict.get('DT').unit,
                            descr='Lfilter applied to DT curve')

        if 'RHOB' in self.keys():
            # Get RHOB curve data
            rhob = self['RHOB']

            # Apply lfilter and add filtered curve to log
            rhob_filtered = apply_lfilter(rhob)
            self.append_curve('RHOB', rhob_filtered, unit=self.curvesdict.get('RHOB').unit,
                            descr='Lfilter applied to RHOB curve')



    def write(self, file_path, version = 2.0, wrap = False,
              STRT = None, STOP = None, STEP = None, fmt = '%10.6g', len_numeric_field=15,
                              header_width=80, data_section_header="~A", mnemonics_header=True):
        """
        Writes to las file, and overwrites if file exisits. Uses parent
        class LASFile.write method with specified defaults.

        Parameters
        ----------
        file_path : str
            path to new las file.
        version : {1.2 or 2} (default 2)
            Version for las file
        wrap : {True, False, None} (default False)
            Specify to wrap data. If None, uses setting from when
            file was read.
        STRT : float (default None)
            Optional override to automatic calculation using the first
            index curve value.
        STOP : float (default None)
            Optional override to automatic calculation using the last
            index curve value.
        STEP : float (default None)
            Optional override to automatic calculation using the first
            step size in the index curve.
        fmt : str (default '%10.5g')
            Format string for numerical data being written to data
            section.

        Example
        -------
        >>> import petropy as ptr
        >>> # reads sample Wolfcamp Log from las file
        >>> log = ptr.log_data('WFMP')
        >>> # define file path to save log
        >>> p = 'path/to/new_file.las'
        >>> log.write(p)

        """
        
        with open(file_path, 'w') as f:
            super(Log, self).write(f, version = version, wrap = wrap,
                                   STRT = STRT, STOP = STOP,
                                   STEP = None, fmt = fmt, len_numeric_field=len_numeric_field,
                              header_width=header_width, data_section_header=data_section_header, mnemonics_header=mnemonics_header)
