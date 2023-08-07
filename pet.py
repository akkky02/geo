import os
import re
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import datetime as dt
from scipy.optimize import nnls
from scipy.signal import lfilter, filtfilt

from lasio import LASFile, CurveItem

"""
Log contains parent classes to work with log data.

    The Log class is subclassed from lasio LASFile class, which provide a
    data structure. The methods are for petrophysical calculations and for
    viewing data with the LogViewer class.

"""


class Log(LASFile):
    """
    Log:

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
        Preconditions log curve by aliasing names.:

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
        standard_curves = ['GR', 'NPHI', 'RHOB', 'ILD', 'PE', 'DT']
        self_keys = list(self.keys())
        for curve in self_keys:
            if curve not in standard_curves and curve!= 'DEPT':
                self.delete_curve(curve)

        # Check if any curves from standard_curves are present after filtering
        remaining_curves = [curve for curve in self.keys() if curve in standard_curves]
        if not remaining_curves:
            raise ValueError("None of the curves from the standard_curves list are present in the data.")

        def apply_lfilter(curve_data):
            valid_mask = ~np.logical_or(np.isnan(curve_data), curve_data == self.well.NULL.value)
            filtered_data = np.empty_like(curve_data)
            filtered_data[valid_mask] = filtfilt(np.ones(n) / n, 1, curve_data[valid_mask], padtype= None)
            # Round the filtered data to a maximum of 4 decimal places
            filtered_data = np.round(filtered_data, 4)
            return np.where(valid_mask, filtered_data, np.nan)

        if 'DT' in self.keys():
            # Get DT curve data
            dt = self['DT']

            # Apply lfilter and add filtered curve to log
            dt_filtered = np.clip(apply_lfilter(dt),a_min=None,a_max=122)
            self['DT'] = dt_filtered
            self.curvesdict.get('DT').descr = 'Lfilter applied to DT curve' 
            # self.append_curve('DT', dt_filtered, unit=self.curvesdict.get('DT').unit,
            #                     descr='Lfilter applied to DT curve')

        if 'RHOB' in self.keys():
            # Get RHOB curve data
            rhob = self['RHOB']

            # Apply lfilter and add filtered curve to log
            rhob_filtered = np.clip(apply_lfilter(rhob),a_min=2.2,a_max=None)
            self['RHOB'] = rhob_filtered
            self.curvesdict.get('RHOB').descr = 'Lfilter applied to RHOB curve'
            # self.append_curve('RHOB', rhob_filtered, unit=self.curvesdict.get('RHOB').unit,
            #                     descr='Lfilter applied to RHOB curve')

    def fluid_properties(self, top=0, bottom=100000, mast=67,
                        temp_grad=0.015, press_grad=0.5, rws=0.1, rwt=70,
                        rmfs=0.4, rmft=100, gas_grav=0.67, oil_api=38, p_sep=100,
                        t_sep=100, yn2=0, yco2=0, yh2s=0, yh20=0, rs=0,
                        lith_grad=1.03, biot=0.8, pr=0.25):
        """
        Calculates fluid properties along wellbore.

            The output add the following calculated curves at each depth:

            PORE_PRESS : (psi)
                Reservoir pore pressure
            RES_TEMP : (°F)
                Reservoir temperature
            NES : (psi)
                Reservoir net effective stress
            RW : (ohm.m)
                Resistivity of water
            RMF : (ohm.m)
                Resistivity of mud filtrate
            RHO_HC : (g / cc)
                Density of hydrocarbon
            RHO_W : (g / cc)
                Density of formation water
            RHO_MF : (g / cc)
                Density of mud filtrate
            NPHI_HC
                Neutron log response of hydrocarbon
            NPHI_W
                Neutron log response of water
            NPHI_MF
                Neutron log response of mud filtrate
            MU_HC : (cP)
                Viscosity of hydrocarbon
            Z
                Compressiblity factor for non-ideal gas.
                Only output if oil_api = 0
            CG : (1 / psi)
                Gas Compressiblity. Only output if oil_api = 0
            BG
                Gas formation volume factor. Only output if oil_api = 0
            BP : (psi)
                Bubble point. Only output if oil_api > 0
            BO
                Oil formation volume factor. Only output if oil_api > 0

            Parameters
            ----------
            top : float (default 0)
                The top depth to begin fluid properties calculation. If
                value is not specified, the calculations will start at
                the top of the log.
            bottom : float (default 100,000)
                The bottom depth to end fluid properties, inclusive. If the
                value is not specified, the calcuations will go to the
                end of the log.
            mast : float (default 67)
                The mean annual surface temperature at the location of the
                well in degrees Fahrenheit.
            temp_grad : float (default 0.015)
                The temperature gradient of the reservoir in °F / ft.
            press_grad : float (default 0.5)
                The pressure gradient of the reservoir in psi / ft.
            rws : float (default 0.1)
                The resistivity of water at surface conditions in ohm.m.
            rwt : float (default 70)
                The temperature of the rws measurement in °F.
            rmfs : float (default 0.4)
                The resistivity of mud fultrate at surface conditions in
                ohm.m
            rmft : float (default 100)
                The temperature of the rmfs measurement in °F
            gas_grav : float (default 0.67)
                The specific gravity of the separator gas. Air = 1,
                CH4 = 0.577
            oil_api : float (default 38)
                The api gravity of oil after the separator
                If fluid system is dry gas, use oil_api = 0.
            p_sep : float (default 100)
                The pressure of the separator, assuming a 2 stage system
                Only used when oil_api is > 0 (not dry gas).
            t_sep : float
                The temperature of the separator, assuming a 2 stage system
                Only used with :code:`oil_api > 0`.
            yn2 : float (default 0)
                Molar fraction of nitrogren in gas.
            yco2 : float (default 0)
                Molar fration of carbon dioxide in gas.
            yh2s : float (default 0)
                Molar fraction of hydrogen sulfide in gas.
            yh20 : float (default 0)
                Molar fraction of water in gas.
            rs : float (default 0)
                Solution gas oil ratio at reservoir conditions.
                If unknwon, use 0 and correlation will be used.
            lith_grad : float (default 1.03)
                Lithostatic overburden pressure gradient in psi / ft.
            biot : float (default 0.8)
                Biot constant.
            pr : float (default 0.25)
                Poissons ratio

            Note
            ----
            Current single phase fluid properties assumes either:

                1. Dry Gas at Reservoir Conditions
                Methane as hydrocarbon type with options to include N2,
                CO2, H2S, or H2O. To assume dry_gas, set
                :code:`oil_api = 0`

                2. Oil at Reservoir Conditions
                Assumes reservoir fluids are either a black or volatile
                oil. Separator conditions of gas are used to calculate
                bubble point and the reservoir fluid properties of the
                reconstituted oil.

            References
            ----------
            Ahmed, Tarek H. Reservoir Engineering Handbook. Oxford: Gulf
                Professional, 2006.

            Lee, John, and Robert A. Wattenbarger. Gas Reservoir
                Engineering. Richardson, TX: Henry L. Doherty Memorial
                Fund of AIME, Society of Petroleum Engineers, 2008.

            Example
            -------
            >>> import petropy as ptr
            >>> from petropy import datasets
            >>> # reads sample Wolfcamp Log from las file
            >>> log = ptr.log_data('WFMP')
            >>> # calculates fluid properties with default settings
            >>> log.fluid_properties()

            See Also
            --------
            :meth:`petropy.Log.fluid_properties_parameters_from_csv`
                loads properties from preconfigured csv file
            :meth:`petropy.Log.multimineral_model`
                builds on fluid properties to calculate full petrophysical
                model

        """

        depths = self['DEPT']
        depth_index = np.logical_and(depths >= top, depths < bottom)
        depths = depths[depth_index]

        form_temp = mast + temp_grad * depths
        pore_press = press_grad * depths

        ### water properties ###
        rw = (rwt + 6.77) / (form_temp + 6.77) * rws
        rmf = (rmft + 6.77) / (form_temp + 6.77) * rmfs

        rw68 = (rwt + 6.77) / (68 + 6.77) * rws
        rmf68 = (rmft + 6.77) / (68 + 6.77) * rws

        ### weight percent total dissolved solids ###
        xsaltw = 10 ** (-0.5268 * (np.log10(rw68)) ** 3 - 1.0199 * \
                    (np.log10(rw68)) ** 2 - 1.6693 * (np.log10(rw68)) - 0.3087)
        xsaltmf = 10 ** (-0.5268 * (np.log10(rmf68)) ** 3 - 1.0199 * \
                        (np.log10(rmf68)) ** 2 - 1.6693 * (np.log10(rmf68)) - 0.3087)

        ### bw for reservoir water. ###
        ### Eq 1.83 - 1.85 Gas Reservoir Engineering ###
        dvwt = -1.0001 * 10 ** -2 + 1.33391 * 10 ** -4 * form_temp + \
            5.50654 * 10 ** -7 * form_temp ** 2

        dvwp = -1.95301 * 10 ** -9 * pore_press * form_temp - \
            1.72834 * 10 ** -13 * pore_press ** 2 * form_temp - \
            3.58922 * 10 ** -7 * pore_press - \
            2.25341 * 10 ** -10 * pore_press ** 2

        bw = (1 + dvwt) * (1 + dvwp)

        ### calculate solution gas in water ratio ###
        ### Eq. 1.86 - 1.91 Gas Reservoir Engineering ###
        rsa = 8.15839 - 6.12265 * 10 ** -2 * form_temp + \
            1.91663 * 10 ** -4 * form_temp ** 2 - \
            2.1654 * 10 ** -7 * form_temp ** 3

        rsb = 1.01021 * 10 ** -2 - 7.44241 * 10 ** -5 * form_temp + \
            3.05553 * 10 ** -7 * form_temp ** 2 - \
            2.94883 * 10 ** -10 * form_temp ** 3

        rsc = -1.0 * 10 ** -7 * (9.02505 - 0.130237 * form_temp + \
                                8.53425 * 10 ** -4 * form_temp ** 2 - 2.34122 * 10 ** -6 * \
                                form_temp ** 3 + 2.37049 * 10 ** -9 * form_temp ** 4)

        rswp = rsa + rsb * pore_press + rsc * pore_press ** 2
        rsw = rswp * 10 ** (-0.0840655 * xsaltw * form_temp ** -0.285584)

        ### log responses ###
        rho_w = (2.7512 * 10 ** -5 * xsaltw +
                6.9159 * 10 ** -3 * xsaltw + 1.0005) * bw

        rho_mf = (2.7512 * 10 ** -5 * xsaltmf +
                6.9159 * 10 ** -3 * xsaltmf + 1.0005) * bw

        nphi_w = 1 + 0.4 * (xsaltw / 100)
        nphi_mf = 1 + 0.4 * (xsaltmf / 100)

        ### net effective stress ###
        nes = (((lith_grad * depths) - (biot * press_grad * depths) +
                2 * (pr / (1 - pr)) * (lith_grad * depths) -
                (biot * press_grad * depths))) / 3

        ### gas reservoir ###
        if oil_api == 0:
            # hydrocarbon gravity only
            hc_grav = (gas_grav - 1.1767 * yh2s - 1.5196 * yco2 - \
                    0.9672 * yn2 - 0.622 * yh20) / \
                    (1.0 - yn2 - yco2 - yh20 - yh2s)

            # pseudocritical properties of hydrocarbon
            ppc_h = 756.8 - 131.0 * hc_grav - 3.6 * (hc_grav ** 2)
            tpc_h = 169.2 + 349.5 * hc_grav - 74.0 * (hc_grav ** 2)

            # pseudocritical properties of mixture
            ppc = (1.0 - yh2s - yco2 - yn2 - yh20) * ppc_h + \
                1306.0 * yh2s + 1071.0 * yco2 + \
                493.1 * yn2 + 3200.1 * yh20

            tpc = (1.0 - yh2s - yco2 - yn2 - yh20) * tpc_h + \
                672.35 * yh2s + 547.58 * yco2 + \
                227.16 * yn2 + 1164.9 * yh20

            # Wichert-Aziz correction for H2S and CO2
            if yco2 > 0 or yh2s > 0:
                epsilon = 120 * ((yco2 + yh2s) ** 0.9 -
                                (yco2 + yh2s) ** 1.6) + \
                        15 * (yh2s ** 0.5 - yh2s ** 4)

                tpc_temp = tpc - epsilon
                ppc = (ppc_a * tpc_temp) / \
                    (tpc + (yh2s * (1.0 - yh2s) * epsilon))

                tpc = tpc_temp
            # Casey correction for nitrogen and water vapor
            if yn2 > 0 or yh20 > 0:
                tpc_cor = -246.1 * yn2 + 400 * yh20
                ppc_cor = -162.0 * yn2 + 1270.0 * yh20
                tpc = (tpc - 227.2 * yn2 - 1165.0 * yh20) / \
                    (1.0 - yn2 - yh20) + tpc_cor

                ppc = (ppc - 493.1 * yn2 - 3200.0 * yh20) / \
                    (1.0 - yn2 - yh20) + ppc_cor

            # Reduced pseudocritical properties
            tpr = (form_temp + 459.67) / tpc
            ppr = pore_press / ppc

            ### z factor from Dranchuk and Abou-Kassem fit of ###
            ### Standing and Katz chart ###
            a = [0.3265,
                -1.07,
                -0.5339,
                0.01569,
                -0.05165,
                0.5475,
                -0.7361,
                0.1844,
                0.1056,
                0.6134,
                0.721]

            t2 = a[0] * tpr + a[1] + a[2] / (tpr ** 2) + \
                a[3] / (tpr ** 3) + a[4] / (tpr ** 4)

            t3 = a[5] * tpr + a[6] + a[7] / tpr
            t4 = -a[8] * (a[6] + a[7] / tpr)
            t5 = a[9] / (tpr ** 2)

            r = 0.27 * ppr / tpr
            z = 0.27 * ppr / tpr / r

            counter = 0
            diff = 1
            while counter <= 10 and diff > 10 ** -5:
                counter += 1

                f = r * (tpr + t2 * r + t3 * r ** 2 + t4 * r ** 5 + \
                        t5 * r ** 2 * (1 + a[10] * r ** 2) * \
                        np.exp(-a[10] * r ** 2)) - 0.27 * ppr

                fp = tpr + 2 * t2 * r + 3 * t3 * r ** 2 + \
                    6 * t4 * r ** 5 + t5 * r ** 2 * \
                    np.exp(-a[10] * r ** 2) * \
                    (3 + a[10] * r ** 2 * (3 - 2 * a[10] * r ** 2))

                r = r - f / fp
                diff = np.abs(z - (0.27 * ppr / tpr / r)).max()
                z = 0.27 * ppr / tpr / r

            ### gas compressibility from Dranchuk and Abau-Kassem ###
            cpr = tpr * z / ppr / fp
            cg = cpr / ppc

            ### gas expansion factor ###
            bg = (0.0282793 * z * (form_temp + 459.67)) / pore_press

            ### gas density Eq 1.64 GRE ###
            rho_hc = 1.495 * 10 ** -3 * (pore_press * (gas_grav)) / \
                    (z * (form_temp + 459.67))
            nphi_hc = 2.17 * rho_hc

            ### gas viscosity Lee Gonzalez Eakin method ###
            ### Eqs. 1.63-1.67 GRE ###
            k = ((9.379 + 0.01607 * (28.9625 * gas_grav)) * \
                (form_temp + 459.67) ** 1.5) / \
                (209.2 + 19.26 * (28.9625 * gas_grav) + \
                (form_temp + 459.67))

            x = 3.448 + 986.4 / \
                (form_temp + 459.67) + 0.01009 * (28.9625 * gas_grav)

            y = 2.447 - 0.2224 * x
            mu_hc = 10 ** -4 * k * np.exp(x * rho_hc ** y)

            ### oil reservoir ###
        else:

            # Normalize gas gravity to separator pressure of 100 psi
            ygs100 = gas_grav * (1 + 5.912 * 0.00001 * oil_api * \
                                (t_sep - 459.67) * np.log10(p_sep / 114.7))

            if oil_api < 30:
                if rs == 0 or rs is None:
                    rs = 0.0362 * ygs100 * pore_press ** 1.0937 * \
                        np.exp((25.724 * oil_api) / (form_temp + 459.67))

                bp = ((56.18 * rs / ygs100) * 10 ** \
                    (-10.393 * oil_api / (form_temp + 459.67))) ** 0.84246
                ### gas saturated bubble-point ###
                bo = 1 + 4.677 * 10 ** -4 * rs + 1.751 * 10 ** -5 * \
                    (form_temp - 60) * (oil_api / ygs100) - \
                    1.811 * 10 ** -8 * rs * \
                    (form_temp - 60) * (oil_api / ygs100)
            else:
                if rs == 0 or rs is None:
                    rs = 0.0178 * ygs100 * pore_press ** 1.187 * \
                        np.exp((23.931 * oil_api) / (form_temp + 459.67))

                bp = ((56.18 * rs / ygs100) * 10 ** \
                    (-10.393 * oil_api / (form_temp + 459.67))) ** 0.84246

                ### gas saturated bubble-point ###
                bo = 1 + 4.670 * 10 ** -4 * rs + 1.1 * \
                    10 ** -5 * (form_temp - 60) * (oil_api / ygs100) + \
                    1.337 * 10 ** -9 * rs * (form_temp - 60) * \
                    (oil_api / ygs100)

            ### calculate bo for undersaturated oil ###
            pp_gt_bp = np.where(pore_press > bp + 100)[0]
            if len(pp_gt_bp) > 0:
                bo[pp_gt_bp] = bo[pp_gt_bp] * np.exp(-(0.00001 * \
                                                    (-1433 + 5 * rs + 17.2 * form_temp[pp_gt_bp] - \
                                                        1180 * ygs100 + 12.61 * oil_api)) * \
                                                    np.log(pore_press[pp_gt_bp] / bp[pp_gt_bp]))

            ### oil properties ###
            rho_hc = (((141.5 / (oil_api + 131.5) * 62.428) + \
                    0.0136 * rs * ygs100) / bo) / 62.428
            nphi_hc = 1.003 * rho_hc

            ### oil viscosity from Beggs-Robinson ###
            ### RE Handbook Eqs. 2.121 ###
            muod = 10 ** (np.exp(6.9824 - 0.04658 * oil_api) * \
                        form_temp ** -1.163) - 1

            mu_hc = (10.715 * (rs + 100) ** -0.515) * \
                    muod ** (5.44 * (rs + 150) ** -0.338)

            ### undersaturated oil viscosity from Vasquez and Beggs ###
            ### Eqs. 2.123 ###
            if len(pp_gt_bp) > 0:
                mu_hc[pp_gt_bp] = mu_hc[pp_gt_bp] * \
                                (pore_press[pp_gt_bp] / bp[pp_gt_bp]) ** \
                                (2.6 * pore_press[pp_gt_bp] ** 1.187 * \
                                10 ** (-0.000039 * pore_press[pp_gt_bp] - 5))

            output_curves = [
                {'mnemonic': 'PORE_PRESS', 'data': pore_press, 'unit': 'psi',
                'descr': 'Calculated Pore Pressure'},

                {'mnemonic': 'RES_TEMP', 'data': form_temp, 'unit': 'F',
                'descr': 'Calculated Reservoir Temperature'},

                {'mnemonic': 'NES', 'data': nes, 'unit': 'psi',
                'descr': 'Calculated Net Effective Stress'},

                {'mnemonic': 'RW', 'data': rw, 'unit': 'ohmm',
                'descr': 'Calculated Resistivity Water'},

                {'mnemonic': 'RMF', 'data': rmf, 'unit': 'ohmm',
                'descr': 'Calculated Resistivity Mud Filtrate'},

                {'mnemonic': 'RHO_HC', 'data': rho_hc, 'unit': 'g/cc',
                'descr': 'Calculated Density of Hydrocarbon'},

                {'mnemonic': 'RHO_W', 'data': rho_w, 'unit': 'g/cc',
                'descr': 'Calculated Density of Water'},

                {'mnemonic': 'RHO_MF', 'data': rho_mf, 'unit': 'g/cc',
                'descr': 'Calculated Density of Mud Filtrate'},

                {'mnemonic': 'NPHI_HC', 'data': nphi_hc, 'unit': 'v/v',
                'descr': 'Calculated Neutron Log Response of Hydrocarbon'},

                {'mnemonic': 'NPHI_W', 'data': nphi_w, 'unit': 'v/v',
                'descr': 'Calculated Neutron Log Response of Water'},

                {'mnemonic': 'NPHI_MF', 'data': nphi_mf, 'unit': 'v/v',
                'descr': 'Calculated Neutron Log Response of Mud Filtrate'},

                {'mnemonic': 'MU_HC', 'data': mu_hc, 'unit': 'cP',
                'descr': 'Calculated Viscosity of Hydrocarbon'}
            ]

            for curve in output_curves:
                if curve['mnemonic'] in self.keys():
                    self[curve['mnemonic']][depth_index] = curve['data']
                else:
                    data = np.empty(len(self[0]))
                    data[:] = np.nan
                    data[depth_index] = curve['data']
                    curve['data'] = data
                    self.append_curve(curve['mnemonic'], data=curve['data'],
                                    unit=curve['unit'], descr=curve['descr'])

            ### gas curves ###
            if oil_api == 0:
                gas_curves = [
                    {'mnemonic': 'Z', 'data': z, 'unit': '',
                    'descr': 'Calculated Real Gas Z Factor'},

                    {'mnemonic': 'CG', 'data': cg, 'unit': '1 / psi',
                    'descr': 'Calculated Gas Compressibility'},

                    {'mnemonic': 'BG', 'data': bg, 'unit': '',
                    'descr': 'Calculated Gas Formation Volume Factor'}
                ]

                for curve in gas_curves:
                    if curve['mnemonic'] in self.keys():
                        self[curve['mnemonic']][depth_index] = curve['data']
                    else:
                        data = np.empty(len(self[0]))
                        data[:] = np.nan
                        data[depth_index] = curve['data']
                        curve['data'] = data
                        self.append_curve(curve['mnemonic'],
                                        data=curve['data'],
                                        unit=curve['unit'],
                                        descr=curve['descr'])

            ### oil curves ###
            else:
                oil_curves = [
                    {'mnemonic': 'BO', 'data': bo, 'unit': '',
                    'descr': 'Calculated Oil Formation Volume Factor'},

                    {'mnemonic': 'BP', 'data': bp, 'unit': 'psi',
                    'descr': 'Calculated Bubble Point'}
                ]

                for curve in oil_curves:
                    if curve['mnemonic'] in self.keys():
                        self[curve['mnemonic']][depth_index] = curve['data']
                    else:
                        data = np.empty(len(self[0]))
                        data[:] = np.nan
                        data[depth_index] = curve['data']
                        curve['data'] = data
                        self.append_curve(curve['mnemonic'],
                                        data=curve['data'],
                                        unit=curve['unit'],
                                        descr=curve['descr'])

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
