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


    def precondition(self, drho_matrix = 2.71):
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
        ALIAS_XML_PATH = os.path.join(file_dir, 'data', 'MCG_ALIAS_EXPORT.xml')

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

        if 'RHOB' not in self.keys() and 'DPHI' in self.keys():
            calculated_rho = np.empty(len(self[0]))
            non_null_depth_index=np.where(~np.isnan(self['DPHI']))[0]
            non_null_depths = self['DPHI'][non_null_depth_index]
            calculated_rho[non_null_depth_index] = \
                      drho_matrix - (drho_matrix - 1) * non_null_depths

            self.append_curve('RHOB', calculated_rho, unit = 'g/cc',
                       value = '',
                       descr = 'Calculated bulk density from density \
                               porosity assuming rho matrix = %.2f' % \
                               drho_matrix)
