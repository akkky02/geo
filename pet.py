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

        for curve in remaining_curves:
            if curve == 'DEPT':
                continue
            self[curve] = np.round(np.clip(self[curve], a_min=0.0001, a_max=None), 4)

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

    def multimineral_model(self, top = 0, bottom = 100000,
    gr_matrix = 10, nphi_matrix = 0, gr_clay = 350, rho_clay = 2.64,
    nphi_clay = 0.65, pe_clay = 4, rma = 180, rt_clay = 80,
    vclay_linear_weight = 1, vclay_clavier_weight = 0.5,
    vclay_larionov_weight = 0.5, vclay_nphi_weight = 1,
    vclay_nphi_rhob_weight = 1, vclay_cutoff = 0.1, rho_om = 1.15,
    nphi_om = 0.6, pe_om = 0.2, ro = 1.6, lang_press = 670,
    passey_nphi_weight = 1, passey_rhob_weight = 1, passey_lom = 10,
    passey_baseline_res = 40, passey_baseline_rhob = 2.65,
    passey_baseline_nphi = 0, schmoker_weight = 1,
    schmoker_slope =  0.7257, schmoker_baseline_rhob = 2.6,
    rho_pyr = 5, nphi_pyr = 0.13, pe_pyr = 13, om_pyrite_slope = 0.2,
    include_qtz = 'YES', rho_qtz = 2.65, nphi_qtz = -0.04,
    pe_qtz = 1.81, include_clc = 'YES', rho_clc = 1.71, nphi_clc = 0,
    pe_clc = 5.08, include_dol = 'YES', rho_dol = 2.85,
    nphi_dol = 0.04, pe_dol = 3.14, include_x = 'NO',
    name_x = 'Gypsum', name_log_x = 'GYP', rho_x = 2.35,
    nphi_x = 0.507, pe_x = 4.04, pe_fl = 0, m = 2, n = 2, a = 1,
    archie_weight = 0, indonesia_weight = 1, simandoux_weight = 0,
    modified_simandoux_weight = 0, waxman_smits_weight = 0, cec = -1,
    buckles_parameter = -1):
        """
        Calculates a petrophysical lithology and porosity model for
        conventional and unconventional reservoirs. For each depth, the
        method iterates in a 4 step loop until convergence.

            **1. Calculate Clay Volume**

            Clay volume is calculated using a weighted method. Five
            different equations are available

            I. Linear

            .. math::

                gr\_index &= \\frac{GR_LOG - gr\_matrix}
                                {gr\_clay - gr_matrix}

                VCLAY &= gr\_index

            II. Clavier

            .. math::

                VCLAY = 1.7 - \sqrt{3.38 - (gr\_index + 0.7)^2}

            III. Larionov Tertiary Rocks

            .. math::

                VCLAY = 0.083 * (2^{3.7 * gr\_index} - 1)

            IV. Neutron
            Calculate apparent nuetron log without organic matter, nphia.

            .. math::

                nphia = NPHI\_LOG + (nphi\_matrix - nphi\_om) * vom

            Calculate vclay using neutron

            .. math::

                VCLAY = \\frac{nphia - nphi\_matrix}
                            {nphi\_clay - nphi\_matrix}

            V. Neutron Density

            Calculate apprent density log without organic mater, rhoba.

            .. math::

                rhoba = RHOB\_LOG + (rhom - rho\_om) * vom

            Calculate vclay using neutron density

            .. math::

                m1 &= \\frac{nphi\_fl - nphi\_matrix}{rho\_fl - rhom} \\\\
                x1 &= nphi + m1 * (rhom - rhoba) \\\\
                x2 &= nphi\_clay + m1 * (rhom - rhoba) \\\\
                VCLAY &= \\frac{x1 - nphi\_matrix}{x2 - nphi\_matrix} \\\\


            First, the clay volume of the resepective euqations are
            calculated. They are then weighted with the vclay_weight.
            For example, if vclay_weight for every method is 1, then the
            final vclay is the average of the four equations. To use a
            single method, set :code:`vclay_method_weight = 1` and all
            other :code:`vclay_weight = 0`.

            **2. Calculate Total Organic Carbon And Pyrite**

            TOC is calculated use a weighted method like vclay with three
            available equations:

            I. Schomker's Density Correlation

            .. math::

                TOC = schmoker\_slope*(schmoker\_baseline\_rhob-RHOB\_LOG)

            II. Passey's Nuetron Delta Log R

            .. math::

                dlr &= 10^{\\frac{ILD\_LOG}{passey\_baseline\_res}} +
                    4 * (NPHI\_LOG - passey\_baseline\_nphi)

                TOC&=\\frac{dlr\_nphi * 10^{2.297 - 0.1688 * passey\_lom}}
                        {100}

            III. Passey's Density Delta Log R

            .. math::

                dlr &= 10^{\\frac{ILD\_LOG}{passey\_baseline\_res}} -
                    2.5 * (RHOB\_LOG - passey\_baseline\_rhob)

                TOC&=\\frac{dlr\_nphi * 10^{2.297 - 0.1688 * passey\_lom}}
                        {100}

            For conventional reservoirs without organics,
            set :code:`vclay_cutoff = 1`.

            **3. Calculate Minerals And Porosity**

            Non-negative least squares is used to find the remaining
            minerals according to the method described in Chapter 4 of
            Doveton's Principles of Mathematical Petrophysics.

            .. math::

                V = C^{-1} L

            Pure mineral log responses are required. For example, quartz
            would have the input:
            ::

                include_qtz = 'YES'
                rho_qtz = 2.65
                nphi_qtz = -0.04
                pe_qtz = 1.81

            An option to include exotic minerals is by specifiying the
            density, neutron, and pe response of mineral 'X'. For example,
            to add gypsum, use these parameters:
            ::

                include_x = 'YES'
                name_x = 'Gypsum'
                name_log_x = 'GYP'
                rho_x = 2.35
                nphi_x = 0.507
                pe_x = 4.04

            To exclude minerals because they are not present or essentially
            not present in the reservoir, set the include parameter to
            'NO'. For example, to exclude dolomite:
            ::

                include_dol = 'NO'

            4. Calculate Saturations

            Saturation is calculated using a weighted method like vclay and
            toc. To use a single equation set equation_weight = 1 and all
            other equation_weight = 0. For example, to use only the
            Indonesia equation set parameters to:
            ::

                archie_weight = 0
                simandoux_weight = 0
                modified_simandoux_weight = 0
                indonesia_weight = 1
                waxman_smits_weight = 0

            Five saturation equations are available

            I. Archie

            .. math::

                SW=\left(\\frac{a \\times RW\_LOG}
                            {ILD\_LOG \\times phie^m}
                    \\right)^{\\frac{1}{n}}

            II. Simandoux

            .. math::

                c &= \\frac{(1 - vclay) \\times a \\times ILD\_LOG}
                        {phie^m}

                d &= \\frac{c \\times vclay}{2 \\times rt\_clay}

                e &= \\frac{c}{ILD\_LOG}

                SW &= ((d^2 + e)^2 - d)^{\\frac{2}{n}}

            III. Modified Simandoux

            IV. Indonesia (Poupon-Leveaux)

            .. math::

                f &= \sqrt{\\frac{phie^m}{ILD\_LOG}} \\\\
                g &= \sqrt{\\frac{vclay^{2 - vclay}}{rt\_clay}} \\\\
                SW &= ((f + g)^2 * ILD\_LOG)^{\\frac{-1}{n}}

            V. Waxman And Smits

            CEC
                if cec <= 0

                .. math::

                    cec = 10^{1.9832 \\times vclay - 2.4473}

                else
                    use input cec

            SW

            .. math::

                rw77&=ILD\_LOG * \\frac{reservoir\_temperature + 6.8}
                                        {83.8}

                b &= 4.6 * (1 - 0.6 \\times e^{\\frac{-0.77}{rw77}})

                f &= \\frac{a}{phie^m}

                qv &= \\frac{cec (1 - phie) rhom}{phie}

                SW &= 0.5 * \left((-b \\times qv \\times rw77) +
                    \sqrt{(b \\times qv \\times rw77)^2 +
                    \\frac{4 * f * rw}{ILD\_LOG}}
                    \\right)^{\\frac{2}{n}}

            **4. Update Fluid Properties**

            .. math::

                rho\_fl &= RHO\_W \\times Sw + RHO\_HC \\times (1 - Sw)

                nphi\_fl &= NPHI\_W \\times Sw + NPHI\_HC \\times (1 - Sw)

            Parameters
            ----------
            gr_matrix : float (default 10)
                Gamma Ray response of clean (non-clay) matrix
            nphi_matrix : float (default 0)
                Neutron response of clean (non-clay) matrix
            gr_clay : float (default 450)
                Gamma Ray response of pure clay matrix
            rho_clay : float (default 2.64)
                Density of pure clay matrix
            nphi_clay : float (default 0.65)
                Neutron reponse of pure clay matrix
            pe_clay : float (default 4)
                Photoelectric response of pure clay matrix
            rma : float (default 180)
                Resistivity of clean tight matrix
            rt_clay : float (default 80)
                Resistivity for inorganic shale matrix
            vclay_linear_weight : float (default 1)
                Weight of liner clay volume
            vclay_clavier_weight : float (defaul 0.5)
                Weight of Clavier clay volume
            vclay_larionov_weight : float (defaul 0.5)
                Weight of Larionov clay volume
            vclay_nphi_weight : float (default 1)
                Weight of Neutron clay volume
            vclay_nphi_rhob_weight : float (default 1)
                Weight of Neutron Density clay volume
            vclay_cutoff : float (default 0.05)
                Cutoff for oranics calculation.
                If vclay < vclay_cutoff then toc = 0
            rho_om : float (default 1.15)
                Density of organic matter
            nphi_om : float (default 0.6)
                Neutron response of pure organic matter
            pe_om : float (default 0.2)
                Photoelectric response of pure organic matter
            ro : float (default 1.6)
                Vitronite reflectance of organic matter
            lang_press : float (default 670)
                Langmiur pressure gas adsorption on organic matter in psi
            passey_nphi_weight : float (default 1)
                Weight for Passey nphi toc
            passey_rhob_weight : float (default 1)
                Weight for Passey rhob toc
            passey_lom : float (default 10)
                Passey level of organic maturity
            passey_baseline_res : float (default 40)
                Passey inorganic baseline resistivity
            passey_baseline_rhob : float (default 2.65)
                Passey inorganic baseline density
            passey_baseline_nphi : float (default 0)
                Passey inorganic baseline neutron
            schmoker_weight : float (default 1)
                Weight for Schmoker toc
            schmoker_slope : float (default 0.7257)
                Slope for schmoker density to toc correlation
            schmoker_baseline_rhob : float (default 2.6)
                Density cutoff for schmoker toc correlation
            rho_pyr : float (default 5)
                Density of pyrite
            nphi_pyr : float (default 0.13)
                Neutron response of pure pyrite
            pe_pyr : float (default 13)
                Photoelectric response of pure pyrite
            om_pyrite_slope : float (default 0.2)
                Slope correlating pyrite volume to organic matter
            include_qtz : str {'YES', 'NO'} (default 'YES')
                Toggle to include or exclude qtz.
                'YES' to include. 'NO' to exclude.
            rho_qtz : float (default 2.65)
                Density of quartz
            nphi_qtz : float (default -0.04)
                Neutron response for pure quartz
            pe_qtz : float (default 1.81)
                Photoelectric response for pure quartz
            include_clc : str {'YES', 'NO'} (default 'YES')
                Toggle to include or exclude clc.
                'YES' to include. 'NO' to exclude.
            rho_clc : float (default 2.71)
                Density of calcite
            nphi_clc : float (default 0)
                Neutron response for pure calcite
            pe_clc : float (default 5.08)
                Photoelectric response for pure calcite
            include_dol : str {'YES', 'NO'} (default 'YES')
                Toggle to include or exclude dol.
                'YES' to include. 'NO' to exclude.
            rho_dol : float (default 2.85)
                Density of dolomite
            nphi_dol : float (default 0.04)
                Neutron response to dolomite
            pe_dol : float (default 3.14)
                Photoelectric response to dolomite
            include_x : str {'YES', 'NO'} (default 'NO')
                Toggle to include or exclude exotic mineral, x.
                'YES' to include. 'NO' to exclude.
            name_x : str (default 'Gypsum')
                Name of exotic mineral, x.
            name_log_x : str (default 'GYP')
                Log name of exotic mineral, x
            rho_x : float (default 2.35)
                Density of exotic mineral, x
            nphi_x : float (default 0.507)
                Neutron response of exotic mineral, x
            pe_x : float (default 4.04)
                Photoelectric respone of exotic mineral, x
            pe_fl : float (default 0)
                Photoelectric response of reservoir fluid.
            m : float (default 2)
                Cementation exponent
            n : float (default 2)
                Saturation exponent
            a : float (default 1)
                Cementation constant
            cec : float (default -1)
                Cation Exchange Capaticy for use in Waxman Smits equation.
                If cec = -1, correlation equation is used to calculate cec.
            archie_weight : float (default 0)
                Weight for archie Sw
            indonesia_weight : float (default 1)
                Weight for Indonesia Sw
            simandoux_weight : float (default 0)
                Weight for Simandoux Sw
            modified_simandoux_weight : float (default 0)
                Weight for Modified Simandoux Sw
            waxman_smits_weight : float (default 0)
                Weight for Waxman Smits Sw
            buckles_parameter : float (default -1)
                Buckles parameter for calculating irreducible water
                saturation. If less than 0, it is calculated using a
                correlation.

            Raises
            ------
            ValueError
                If fluid properties curve values are not present in log,
                then ValueError is raised with incorrect curve requirements

            ValueError
                If raw curves GR, NPHI, RHOB, and ILD are not
                present, then ValueError is raised with incorrect curve
                requirements as raw curve is either not present or
                precondtioning has not been properly run.

            ValueError
                If no formation value factor is found, then ValueError is
                raised to satisfy the calculation requirements.

            References
            ----------
            **VCLAY**

            Clavier, C., W. Hoyle, and D. Meunier, 1971a, Quantitative
                interpretation of thermal neutron decay time logs: Part I.
                Fundamentals and techniques: Journal of Petroleum
                Technology, 23, 743–755

            Clavier, C., W. Hoyle, and D. Meunier, 1971b, Quantitative
                interpretation of thermal neutron decay time logs: Part II.
                Interpretation example, interpretation accuracy, and
                timelapse technique: Journal of Petroleum Technology,
                23, 756–763.

            Larionov VV (1969).Borehole Radiometry: Moscow, U.S.S.R. Nedra.

            Nuetron, and Neutron Density taken from presentations.
            Need publish papers for citation.

            **TOC**

            Passey, Q. R., Creaney, S., Kulla, J. B., Moretti, F. J.,
                Stroud, J. D., 1990, Practical Model for Organic Richness
                from Porosity and Resistivity Logs, AAPG Bulletin, 74,
                1777-1794

            Schmoker, J.W., 1979, Determination of organic content of
                Appalachian Devonian shales from formation-density logs:
                American Association of Petroleum Geologists Bulletin,
                v.63, no.9, p.1504-1509

            **MATRIX**

            Doveton, John H. Principles of Mathematical Petrophysics.
                Oxford: Oxford University Press, 2014.

            **SATURATIONS**

            Archie, G.E. 1942. The Electrical Resistivity Log as an Aid in
                Determining Some Reservoir Characteristics. Trans. of AIME
                146 (1): 54-62.

            Bardon, C., and Pied, B.,1969, Formation water saturation in
                shaly sands: Society of Professional Well Log Analysts 10th
                Annual Logging Symposium Transactions: Paper Z,19 pp.

            Poupon, A. and Leveaux, J. 1971. Evaluation of Water
                Saturations in Shaly Formations. The Log Analyst 12 (4)

            Simandoux, P., 1963, Dielectricmeasurements on porous media
                application to the measurement of water saturations: study
                of the behaviour of argillaceous formations: Revue de
                l'Institut Francais du Petrole 18, Supplementary Issue,
                p. 193-215.

            Waxman, M.H., and L.J.M. Smits, Electrical Conductivity in Oil
                Bearing Shaly Sands, Society of Petroleum Engineers
                Journal, June, p.107-122, 1968.


            Note
            ----
                1. Clay bound water
                    Clay bound water is included as part of the clay volume
                    based on the default nphi_clay = 0.65. To calculate
                    clay bound water seperately, set nphi_clay to clay
                    matrix absent clay bound water, and include appropriate
                    buckles_parameter for bound water saturations.

                2. Organics
                    No differieniation is made between kerogen and other
                    organic matter.

            Example
            -------
            >>> import petropy as ptr
            >>> from petropy import datasets
            >>> # reads sample Wolfcamp Log from las file
            >>> log = ptr.log_data('WFMP')
            >>> # calculates fluid properties with default settings
            >>> log.fluid_properties()
            >>> # calculates multimeral model with default settings
            >>> log.multimineral_model()

            See Also
            --------
            :meth:`petropy.Log.formation_multimineral_model`
                uses multimineral_model accross formations

            """

        ### initialize required curves ###
        required_raw_curves = ['GR', 'NPHI', 'RHOB', 'ILD']

        ### check if PE is availble ###
        if 'PE' in self.keys():
            use_pe = True
            required_raw_curves += ['PE']
        else:
            use_pe = False

        ### check for requirements ###
        for curve in required_raw_curves:
            if curve not in self.keys():
                raise ValueError('Raw curve %s not found and is \
                             required for multimineral_model.' % curve)

        required_curves_from_fluid_properties = ['RW', 'RHO_HC',
                                                'RHO_W', 'NPHI_HC',
                                                'NPHI_W', 'RES_TEMP',
                                                'NES', 'PORE_PRESS']

        for curve in required_curves_from_fluid_properties:
            if curve not in self.keys():
                raise ValueError('Fluid Properties curve %s not found.\
                     Run fluid_properties before multimineral_model.' \
                     % curve)

        all_required_curves = required_raw_curves +\
                              required_curves_from_fluid_properties

        if 'BO' not in self.keys() and 'BG' not in self.keys():
            raise ValueError('Formation Volume Factor required for \
                      multimineral_model. Run fluid_properties first.')

        if 'BO' in self.keys():
            hc_class = 'OIL'
        else:
            hc_class = 'GAS'

        ### initialize minerals ###
        if include_qtz.upper()[0] == 'Y':
            include_qtz = True
        else:
            include_qtz = False

        if include_clc.upper()[0] == 'Y':
            include_clc = True
        else:
            include_clc = False

        if include_dol.upper()[0] == 'Y':
            include_dol = True
        else:
            include_dol = False

        if include_x.upper()[0] == 'Y':
            include_x = True
            name_log_x = name_log_x.upper()
        else:
            include_x = False

        ## check for existence of calculated curves ###
        ### add if not found ##
        nulls = np.empty(len(self[0]))
        nulls[:] = np.nan

        output_curves = [
            {'mnemonic': 'PHIE', 'data': np.copy(nulls), 'unit': 'v/v',
            'descr': 'Effective Porosity'},

            {'mnemonic': 'SW', 'data': np.copy(nulls), 'unit': 'v/v',
            'descr': 'Water Saturation'},

            {'mnemonic': 'SHC', 'data': np.copy(nulls), 'unit': 'v/v',
            'descr': 'Hydrocarbon Saturation'},

            {'mnemonic': 'BVH', 'data': np.copy(nulls), 'unit': 'v/v',
            'descr': 'Bulk Volume Hydrocarbon'},

            {'mnemonic': 'BVW', 'data': np.copy(nulls), 'unit': 'v/v',
            'descr': 'Bulk Volume Water'},

            {'mnemonic': 'BVWI', 'data': np.copy(nulls), 'unit': 'v/v',
            'descr': 'Bulk Volume Water Irreducible'},

            {'mnemonic': 'BVWF', 'data': np.copy(nulls), 'unit':
            'v/v', 'descr': 'Bulk Volume Water Free'},

            {'mnemonic': 'BVOM', 'data': np.copy(nulls), 'unit': 'v/v',
            'descr': 'Bulk Volume Fraction Organic Matter'},

            {'mnemonic': 'BVCLAY', 'data': np.copy(nulls), 'unit':'v/v',
            'descr': 'Bulk Volume Fraction Clay'},

            {'mnemonic': 'BVPYR', 'data': np.copy(nulls), 'unit': 'v/v',
            'descr': 'Bulk Volume Fraction Pyrite'},

            {'mnemonic': 'VOM', 'data': np.copy(nulls), 'unit': 'v/v',
            'descr': 'Matrix Volume Fraction Organic Matter'},

            {'mnemonic': 'VCLAY', 'data': np.copy(nulls), 'unit': 'v/v',
            'descr': 'Matrix Volume Fraction Clay'},

            {'mnemonic': 'VPYR', 'data': np.copy(nulls), 'unit': 'v/v',
            'descr': 'Matrix Volume Fraction Pyrite'},

            {'mnemonic': 'RHOM', 'data': np.copy(nulls), 'unit': 'g/cc',
            'descr': 'Matrix Density'},

            {'mnemonic': 'TOC', 'data': np.copy(nulls), 'unit': 'wt/wt',
            'descr': 'Matrix Weight Fraction Organic Matter'},

            {'mnemonic': 'WTCLAY', 'data':np.copy(nulls),'unit':'wt/wt',
            'descr': 'Matrix Weight Fraction Clay'},

            {'mnemonic': 'WTPYR', 'data': np.copy(nulls),'unit':'wt/wt',
            'descr': 'Matrix Weight Fraction Pyrite'},
        ]
        for curve in output_curves:
            if curve['mnemonic'] not in self.keys():
                self.append_curve(curve['mnemonic'], curve['data'],
                               unit = curve['unit'],
                               descr = curve['descr'])

        qtz_curves = [
            {'mnemonic': 'BVQTZ', 'data': np.copy(nulls), 'unit': 'v/v',
            'descr': 'Bulk Volume Fraction Quartz'},
            {'mnemonic': 'VQTZ', 'data': np.copy(nulls), 'unit': 'v/v',
            'descr': 'Matrix Volume Fraction Quartz'},
            {'mnemonic': 'WTQTZ', 'data': np.copy(nulls),'unit':'wt/wt',
            'descr': 'Matrix Weight Fraction Quartz'}
        ]
        if include_qtz:
            for curve in qtz_curves:
                if curve['mnemonic'] not in self.keys():
                    self.append_curve(curve['mnemonic'], curve['data'],
                                   unit = curve['unit'],
                                   descr = curve['descr'])

        clc_curves = [
            {'mnemonic': 'BVCLC', 'data': np.copy(nulls), 'unit': 'v/v',
            'descr': 'Bulk Volume Fraction Calcite'},
            {'mnemonic': 'VCLC', 'data': np.copy(nulls), 'unit': 'v/v',
             'descr': 'Matrix Volume Fraction Calcite'},
            {'mnemonic': 'WTCLC', 'data': np.copy(nulls),'unit':'wt/wt',
            'descr': 'Matrix Weight Fraction Calcite'}
        ]
        if include_clc:
            for curve in clc_curves:
                if curve['mnemonic'] not in self.keys():
                    self.append_curve(curve['mnemonic'], curve['data'],
                                   unit = curve['unit'],
                                   descr = curve['descr'])

        dol_curves = [
            {'mnemonic': 'BVDOL', 'data': np.copy(nulls), 'unit': 'v/v',
            'descr': 'Bulk Volume Fraction Dolomite'},
            {'mnemonic': 'VDOL', 'data': np.copy(nulls), 'unit': 'v/v',
            'descr': 'Matrix Volume Fraction Dolomite'},
            {'mnemonic': 'WTDOL', 'data': np.copy(nulls),'unit':'wt/wt',
            'descr': 'Matrix Weight Fraction Dolomite'}
        ]
        if include_dol:
            for curve in dol_curves:
                if curve['mnemonic'] not in self.keys():
                    self.append_curve(curve['mnemonic'], curve['data'],
                                   unit = curve['unit'],
                                   descr = curve['descr'])

        min_x_curves = [
            {'menmoic': 'V' + name_log_x, 'data': np.copy(nulls),
            'unit': 'v/v', 'descr': 'Bulk Volume Fraction ' + name_x},
            {'mnemonic': 'V' + name_log_x, 'data': np.copy(nulls),
            'unit': 'v/v', 'descr': 'Matrix Volume Fraction '+ name_x},
            {'mnemonic': 'WT' + name_log_x, 'data': np.copy(nulls),
            'unit': 'wt/wt', 'descr': 'Matrix Weight Fraction '+name_x}
        ]
        if include_x:
            for curve in min_x_curves:
                if curve['mnemonic'] not in self.keys():
                    self.append_curve(curve['mnemonic'], curve['data'],
                                   unit = curve['unit'],
                                   descr = curve['descr'])

        oil_curve = {'mnemonic': 'OIP', 'data': np.copy(nulls),
                     'unit': 'Mmbbl / section', 'descr':'Oil in Place'}
        if hc_class == 'OIL':
            if oil_curve['mnemonic'] not in self.keys():
                self.append_curve(oil_curve['mnemonic'], oil_curve['data'],
                               unit = oil_curve['unit'],
                               descr = oil_curve['descr'])

        gas_curves = [
            {'mnemonic': 'GIP', 'data': np.copy(nulls),
            'unit': 'BCF / section', 'descr': 'Gas in Place'},
            {'mnemonic': 'GIP_FREE', 'data': np.copy(nulls),
            'unit': 'BCF / section', 'descr': 'Free Gas in Place'},
            {'mnemonic': 'GIP_ADS', 'data': np.copy(nulls),
            'unit': 'BCF / section', 'descr': 'Adsorbed Gas in Place'}
        ]
        if hc_class == 'GAS':
            for curve in gas_curves:
                if curve['mnemonic'] not in self.keys():
                    self.append_curve(curve['mnemonic'], curve['data'],
                                   unit = curve['unit'],
                                   descr = curve['descr'])

        ### calculations over depths ###
        depth_index = np.intersect1d(np.where(self[0] >= top)[0],
                                     np.where(self[0] < bottom)[0])
        for i in depth_index:

            ### check for null values in data, skip if true ###
            nans = np.isnan([self[x][i] for x in all_required_curves])
            infs = np.isinf([self[x][i] for x in all_required_curves])
            if True in nans or True in infs: continue

            if i > 0:
                sample_rate = abs(self[0][i] - self[0][i - 1])
            else:
                sample_rate = abs(self[0][0] - self[0][1])

            ### initial parameters to start iterations ###
            phie = 0.1
            rhom = 2.68
            rho_fl = 1
            nphi_fl = 1
            vom = 0

            bvqtz_prev = 1
            bvclc_prev = 1
            bvdol_prev = 1
            bvx_prev = 1
            phi_prev = 1
            bvom_prev = 1
            bvclay_prev = 1
            bvpyr_prev = 1

            diff = 1
            counter = 0
            while diff > 1 * 10 ** -3 and counter < 20:
                counter += 1

                ### log curves without organics ###
                rhoba = self['RHOB'][i] + (rhom - rho_om) * vom
                nphia = self['NPHI'][i] + (nphi_matrix - nphi_om)*vom

                ### clay solver ###
                gr_index = np.clip((self['GR'][i] - gr_matrix) \
                           / (gr_clay - gr_matrix), 0, 1)

                ### linear vclay method ###
                vclay_linear = gr_index

                ### Clavier vclay method ###
                vclay_clavier = np.clip(1.7 - np.sqrt(3.38 - \
                                          (gr_index + 0.7) ** 2), 0, 1)

                ### larionov vclay method ###
                vclay_larionov = np.clip(0.083 * \
                                     (2 ** (3.7 * gr_index) - 1), 0, 1)

                # Neutron vclay method without organic correction
                vclay_nphi = np.clip((nphia - nphi_matrix) / \
                                     (nphi_clay - nphi_matrix), 0, 1)

                # Neutron Density vclay method with organic correction
                m1 = (nphi_fl - nphi_matrix) / (rho_fl - rhom)
                x1 = nphia + m1 * (rhom - rhoba)
                x2 = nphi_clay + m1 * (rhom - rho_clay)
                if x2 - nphi_matrix != 0:
                    vclay_nphi_rhob = np.clip((x1 - nphi_matrix) / \
                                              (x2 - nphi_matrix), 0, 1)
                else:
                    vclay_nphi_rhob = 0

                vclay_weights_sum = vclay_linear_weight + \
                       vclay_clavier_weight + vclay_larionov_weight + \
                       vclay_nphi_weight + vclay_nphi_rhob_weight

                vclay = (vclay_linear_weight * vclay_linear + \
                        vclay_clavier_weight * vclay_clavier + \
                        vclay_larionov_weight * vclay_larionov + \
                        vclay_nphi_weight * vclay_nphi + \
                        vclay_nphi_rhob_weight * vclay_nphi_rhob) / \
                        vclay_weights_sum

                vclay = np.clip(vclay, 0, 1)

                bvclay = vclay * (1 - phie)

                ### organics ###
                if vclay > vclay_cutoff:

                    ### Passey ###
                    dlr_nphi = np.log10(self['ILD'][i] / \
                    passey_baseline_res) + 4 * (self['NPHI'][i] - \
                    passey_baseline_nphi)

                    dlr_rhob = np.log10(self['ILD'][i] / \
                    passey_baseline_res) - 2.5 * (self['RHOB'][i] - \
                    passey_baseline_rhob)

                    toc_nphi = np.clip((dlr_nphi * 10 ** (2.297 - \
                                    0.1688 * passey_lom) / 100), 0, 1)

                    toc_rhob = np.clip((dlr_rhob * 10 ** (2.297 - \
                                    0.1688 * passey_lom) / 100), 0, 1)

                    ### Schmoker ###
                    toc_sch = np.clip(schmoker_slope * \
                    (schmoker_baseline_rhob - self['RHOB'][i]), 0, 1)

                    toc_weights = passey_nphi_weight + \
                                  passey_rhob_weight + schmoker_weight

                    ### toc in weight percent ###
                    toc = (passey_nphi_weight * toc_nphi + \
                           passey_rhob_weight * toc_rhob + \
                           schmoker_weight * toc_sch) / toc_weights

                    ### weight percent to volume percent ###
                    volume_om = toc / rho_om

                    # matrix density without organic matter
                    rhom_no_om = (rhom - toc * rho_om) / (1 - toc)

                    # volume of non-organics
                    volume_else = (1 - toc) / rhom_no_om

                    volume_total = volume_om + volume_else

                    vom = volume_om / volume_total
                    bvom = vom * (1 - phie)

                else:
                    toc = 0
                    vom = 0
                    bvom = 0

                ### pyrite correlation with organics ###
                vpyr = np.clip(om_pyrite_slope * vom, 0, 1)
                bvpyr = vpyr * (1 - phie)

                ### create C, V, and L matrix for equations in ###
                ### Chapter 4 of ####
                # Principles of Mathematical Petrophysics by Doveton #

                ### removed effect of clay, organics, and pyrite ###
                volume_unconventional = bvom + bvclay + bvpyr
                rhob_clean = (self['RHOB'][i] - (rho_om * bvom + \
                              rho_clay * bvclay + rho_pyr * bvpyr)) / \
                              (1 - volume_unconventional)

                nphi_clean = (self['NPHI'][i] - (nphi_om * bvom + \
                              nphi_clay*bvclay + nphi_pyr * bvpyr)) / \
                              (1 - volume_unconventional)

                minerals = []
                if use_pe:
                    pe_clean = (self['PE'][i] - (pe_om * bvom + \
                                pe_clay * bvclay + pe_pyr * bvpyr)) / \
                                (1 - bvom - bvclay - bvpyr)

                    l_clean = np.asarray([rhob_clean, nphi_clean,
                                          pe_clean, 1])

                    l = np.asarray([self['RHOB'][i],
                                    self['NPHI'][i],
                                    self['PE'][i], 1])

                    c_clean = np.asarray([0,0,0]) # initialize matrix C

                    if include_qtz:
                        minerals.append('QTZ')
                        mineral_matrix = np.asarray((rho_qtz, nphi_qtz,
                                                     pe_qtz))
                        c_clean = np.vstack((c_clean, mineral_matrix))

                    if include_clc:
                        minerals.append('CLC')
                        mineral_matrix = np.asarray((rho_clc, nphi_clc,
                                                     pe_clc))
                        c_clean = np.vstack((c_clean, mineral_matrix))

                    if include_dol:
                        minerals.append('DOL')
                        mineral_matrix = np.asarray((rho_dol, nphi_dol,
                                                     pe_dol))
                        c_clean = np.vstack((c_clean, mineral_matrix))

                    if include_x:
                        minerals.append('X')
                        mineral_matrix = np.asarray((rho_x, nphi_x,
                                                     pe_x))
                        c_clean = np.vstack((c_clean, mineral_matrix))

                    fluid_matrix = np.asarray((rho_fl, nphi_fl, pe_fl))
                    c_clean = np.vstack((c_clean, fluid_matrix))
                    minerals.append('PHI')

                else:
                    l_clean = np.asarray([rhob_clean, nphi_clean, 1])
                    l = np.asarray([self['RHOB'][i],
                                    self['NPHI'][i],1])

                    c_clean = np.asarray((0,0)) # initialize matrix C

                    if include_qtz:
                        minerals.append('QTZ')
                        mineral_matrix =np.asarray((rho_qtz, nphi_qtz))
                        c_clean = np.vstack((c_clean, mineral_matrix))

                    if include_clc:
                        minerals.append('CLC')
                        mineral_matrix =np.asarray((rho_clc, nphi_clc))
                        c_clean = np.vstack((c_clean, mineral_matrix))

                    if include_dol:
                        minerals.append('DOL')
                        mineral_matrix =np.asarray((rho_dol, nphi_dol))
                        c_clean = np.vstack((c_clean, mineral_matrix))

                    if include_x:
                        minerals.append('X')
                        mineral_matrix = np.asarray((rho_x, nphi_x))
                        c_clean = np.vstack((c_clean, mineral_matrix))

                    fluid_matrix = np.asarray((rho_fl, nphi_fl))
                    c_clean = np.vstack((c_clean, fluid_matrix))
                    minerals.append('PHI')

                c_clean = np.delete(c_clean, 0, 0)

                c_clean = np.vstack((c_clean.T,
                                     np.ones_like(c_clean.T[0])))

                bv_clean = nnls(c_clean, l_clean.T)[0]

                bvqtz = 0
                bvclc = 0
                bvdol = 0
                bvx = 0

                component_sum = np.sum(bv_clean)

                for s, mineral in enumerate(minerals):
                    if mineral == 'QTZ':
                        bvqtz = (bv_clean[s] / component_sum) * \
                                (1 - volume_unconventional)
                        bv_clean[s] = bvqtz
                    if mineral == 'CLC':
                        bvclc = (bv_clean[s] / component_sum) * \
                                (1 - volume_unconventional)
                        bv_clean[s] = bvclc
                    if mineral == 'DOL':
                        bvdol = (bv_clean[s] / component_sum) * \
                                (1 - volume_unconventional)
                        bv_clean[s] = bvdol
                    if mineral == 'X':
                        bvx = (bv_clean[s] / component_sum) * \
                                (1 - volume_unconventional)
                        bv_clean[s] = bvx
                    if mineral == 'PHI':
                        phie = (bv_clean[s] / component_sum) * \
                                (1 - volume_unconventional)
                        bv_clean[s] = phie

                if use_pe:
                    c = np.hstack((c_clean, np.asarray(
                                    (
                                        (rho_om, rho_clay, rho_pyr),
                                        (nphi_om, nphi_clay, nphi_pyr),
                                        (pe_om, pe_clay, pe_pyr),
                                        (1, 1, 1)
                                    )
                                )
                              ))
                else:
                    c = np.hstack((c_clean, np.asarray(
                                    (
                                        (rho_om, rho_clay, rho_pyr),
                                        (nphi_om, nphi_clay, nphi_pyr),
                                        (1, 1, 1))
                                    )
                              ))

                bv = np.append(bv_clean, (bvom, bvclay, bvpyr))

                l_hat = np.dot(c, bv)

                sse = np.dot((l - l_hat).T, l - l_hat)

                prev = np.asarray((bvqtz_prev, bvclc_prev, bvdol_prev,
                                   bvx_prev, phi_prev, bvom_prev,
                                   bvclay_prev, bvpyr_prev))
                cur = np.asarray((bvqtz, bvclc, bvdol, bvx, phie, bvom,
                                  bvclay, bvpyr))

                diff = np.abs(cur - prev).sum()

                bvqtz_prev = bvqtz
                bvclc_prev = bvclc
                bvdol_prev = bvdol
                bvx_prev = bvx
                bvom_prev = bvom
                bvclay_prev = bvclay
                bvpyr_prev = bvpyr
                phi_prev = phie

                avg_percent_error = np.mean(np.abs(l - l_hat) / l) *100

                ### calculate matrix volume fraction ###

                per_matrix = 1 - phie

                vqtz = bvqtz / per_matrix
                vclc = bvclc / per_matrix
                vdol = bvdol / per_matrix
                vx = bvx / per_matrix
                vclay = bvclay / per_matrix
                vom = bvom / per_matrix
                vpyr = bvpyr / per_matrix

        		### calculate weight fraction ###

                mass_qtz = vqtz * rho_qtz
                mass_clc = vclc * rho_clc
                mass_dol = vdol * rho_dol
                mass_x = vx * rho_x
                mass_om = vom * rho_om
                mass_clay = vclay * rho_clay
                mass_pyr = vpyr * rho_pyr

                rhom = mass_qtz + mass_clc + mass_dol + mass_x + \
                       mass_om +mass_clay + mass_pyr

                wtqtz = mass_qtz / rhom
                wtclc = mass_clc / rhom
                wtdol = mass_dol / rhom
                wtx = mass_x / rhom
                wtom = mass_om / rhom
                wtclay = mass_clay / rhom
                wtpyr = mass_pyr / rhom
                toc = wtom

                ### saturations ###

                ### porosity cutoff in case phie =  0 ###
                if phie < 0.001:
                    phis = 0.001
                else:
                    phis = phie

                ### Archie ###
                sw_archie = np.clip(((a * self['RW'][i]) / \
                (self['ILD'][i] * (phis ** m))) ** (1 / n), 0, 1)

                ### Indonesia ###
                sw_ind_a = (phie ** m / self['RW'][i]) ** 0.5
                sw_ind_b = (vclay ** (2.0 - vclay) / rt_clay) ** 0.5
                sw_indonesia = np.clip(((sw_ind_a + sw_ind_b) ** 2.0 *\
                               self['ILD'][i]) ** (-1 / n), 0, 1)

                ### Simandoux ###
                c = (1.0 - vclay) * a * self['RW'][i] / (phis ** m)
                d = c * vclay / (2.0 * rt_clay)
                e = c / self['ILD'][i]
                sw_simandoux = np.clip(((d**2 + e) ** 0.2 - d) ** \
                                                         (2 / n), 0, 1)

                ### modified Simandoux ###
                sw_mod_simd = np.clip((0.5 * self['RW'][i] / \
                                       phis ** m) * ((4 * phis **m) / \
                             (self['RW'][i] * self['ILD'][i]) + \
                             (vclay / rt_clay) ** 2) ** (1 / n) - \
                             vclay / rt_clay, 0, 1)

                ### Waxman Smits ###
                if cec <= 0:
                    cec = 10 ** (1.9832 * vclay - 2.4473)

                rw77 =self['ILD'][i]*(self['RES_TEMP'][i] + 6.8)\
                       / 83.8

                b = 4.6 * (1 - 0.6 * np.exp(-0.77 / rw77))
                f = a / (phis ** m)
                qv = cec * (1 - phis) * rhom / phis
                sw_waxman_smits = np.clip(0.5 * ((-b * qv * rw77) + \
                                              ((b * qv * rw77) ** 2 + \
                                              4 * f * self['RW'][i] / \
                                        self['ILD'][i]) ** 0.5) \
                                            ** (2 / n), 0, 1)

                ### weighted calculation with bv output ###
                weight_saturations = archie_weight + indonesia_weight+\
                       simandoux_weight + modified_simandoux_weight + \
                       waxman_smits_weight

                sw = (archie_weight * sw_archie + \
                      indonesia_weight * sw_indonesia + \
                      simandoux_weight * sw_simandoux + \
                      modified_simandoux_weight * sw_mod_simd + \
                      waxman_smits_weight * sw_waxman_smits) / \
                      weight_saturations

                bvw = phie * sw
                bvh = phie * (1 - sw)

                if hc_class == 'OIL':
                    oip =(7758 * 640 * sample_rate * bvh * 10 ** -6)/ \
                           self['BO'][i] # Mmbbl per sample rate

                elif hc_class == 'GAS':
                    langslope = (-0.08 * self['RES_TEMP'][i] + \
                                 2 * ro + 22.75) / 2
                    gas_ads = langslope * vom * 100 * \
                    (self['PORE_PRESS'][i] / (self['PORE_PRESS'][i] + \
                    lang_press))

                    gip_free=(43560* 640 * sample_rate * bvh *10** -9)\
                                / self['BG'][i]   # BCF per sample rate
                    gip_ads = (1359.7 * 640 * sample_rate * \
                            self['RHOB'][i] * gas_ads * 10 ** -9) / \
                            self['BG'][i]	# BCF per sample rate
                    gip = gip_free + gip_ads

                rho_fl = self['RHO_W'][i] * sw + \
                         self['RHO_HC'][i] * (1 - sw)

                nphi_fl = self['NPHI_W'][i] * sw + \
                          self['NPHI_HC'][i] * (1 - sw)

            ### save calculations to log ###

            ### bulk volume ###
            self['BVOM'][i] = bvom
            self['BVCLAY'][i] = bvclay
            self['BVPYR'][i] = bvpyr

            if include_qtz:
                self['BVQTZ'][i] = bvqtz
            if include_clc:
                self['BVCLC'][i] = bvclc
            if include_dol:
                self['BVDOL'][i] = bvdol
            if include_x:
                self['BV' + name_log_x][i] = bvx

            self['BVH'][i] = bvh
            self['BVW'][i] = bvw

            ### porosity and saturations ###
            self['PHIE'][i] = phie
            self['SW'][i] = sw
            self['SHC'][i] = 1 - sw

            ### mineral volumes ###
            self['VOM'][i] = vom
            self['VCLAY'][i] = vclay
            self['VPYR'][i] = vpyr

            if include_qtz:
                self['VQTZ'][i] = vqtz
            if include_clc:
                self['VCLC'][i] = vclc
            if include_dol:
                self['VDOL'][i] = vdol
            if include_x:
                self['V' + name_log_x] = vx

            ### weight percent ###
            self['RHOM'][i] = rhom
            self['TOC'][i] = toc
            self['WTCLAY'][i] = wtclay
            self['WTPYR'][i] = wtpyr

            if include_qtz:
                self['WTQTZ'][i] = wtqtz
            if include_clc:
                self['WTCLC'][i] = wtclc
            if include_dol:
                self['WTDOL'][i] = wtdol
            if include_x:
                self['WT' + name_log_x] = wtx

            # find irreducible water if buckles_parameter is specified
            if buckles_parameter > 0:
                sw_irr = buckles_parameter / (phie / (1 - vclay))
                bvwi = phie * sw_irr
                bvwf = bvw - bvwi
                self['BVWI'][i] = bvwi
                self['BVWF'][i] = bvwf

            if hc_class == 'OIL':
                self['OIP'][i] = oip

            elif hc_class == 'GAS':
                self['GIP_FREE'][i] = gip_free
                self['GIP_ADS'][i] = gip_ads
                self['GIP'][i] = gip

        ### find irreducible water saturation outside of loop ###
        ### since parameters depend on calculated values ###

        if buckles_parameter < 0:
            buckles_parameter=np.mean(self['PHIE'][depth_index] * \
                                      self['SW'][depth_index])

            ir_denom = (self['PHIE'][depth_index] / \
                       (1 - self['VCLAY'][depth_index]))
            ir_denom[np.where(ir_denom < 0.001)[0]] = 0.001
            sw_irr = buckles_parameter / ir_denom

            self['BVWI'][depth_index] = \
                              self['PHIE'][depth_index] * sw_irr

            self['BVWF'][depth_index] = self['BVW'][depth_index] - \
                                        self['BVWI'][depth_index]



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
