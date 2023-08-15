# -*- coding: utf-8 -*-
"""
Electrofacies is a model to calculate numerical facies from log data.
It uses sckit-learn for standardization and clustering.

"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans

# def electrofacies(logs, formations, curves, n_clusters, log_scale = [],
#                   n_components = 0.85, curve_name = 'FACIES'):
    

def electrofacies(logs, curves = ['GR', 'NPHI', 'RHOB', 'ILD'], n_clusters = 6, log_scale=['ILD'],
                  n_components=0.9, curve_name='FACIES'):
    
    required_curves = ['GR', 'NPHI', 'RHOB', 'ILD']
    optional_curves = ['PE', 'DT']

    missing_curves = [curve for curve in required_curves if curve not in curves]
    if missing_curves:
        raise ValueError(f"Required log curves {missing_curves} are missing for electrofacies.")
    
    available_curves = set(curves).intersection(required_curves + optional_curves)
    
    if not available_curves:
        raise ValueError("No required or optional log curves available in the provided curves list.")

    curves = list(available_curves)


    df = pd.DataFrame()
    
    for log in logs:
        if log.well['UWI'] is None:
            raise ValueError('UWI required for log identification.')

        log_df = log.df()
        log_df['UWI'] = log.well['UWI'].value
        log_df['DEPTH_INDEX'] = np.arange(0, len(log[0]))
        df = pd.concat([df, log_df], ignore_index=False)

    for s in log_scale:
        df[s] = np.log(df[s])

    not_null_rows = pd.notnull(df[curves]).all(axis=1)
    
    X = StandardScaler().fit_transform(df.loc[not_null_rows, curves])

    pc = PCA(n_components=n_components).fit(X)

    components = pd.DataFrame(data=pc.transform(X),
                               index=df[not_null_rows].index)

    minibatch_input = components.to_numpy()

    components.columns = ['PC%i' % x for x in range(1, pc.n_components_ + 1)]
    components['UWI'] = df.loc[not_null_rows, 'UWI']
    components['DEPTH_INDEX'] = df.loc[not_null_rows, 'DEPTH_INDEX']

    size = len(components) // 20
    if size > 100000:
        size = 100000
    elif size < 100:
        size = 100

    df.loc[not_null_rows, curve_name] = MiniBatchKMeans(n_clusters=n_clusters,
                                                        batch_size=size).fit_predict(minibatch_input)
    df.loc[not_null_rows, curve_name] += 1

    for log in logs:
        uwi = log.well['UWI'].value

        for v, vector in enumerate(pc.components_):
            v += 1
            pc_curve = 'PC%i' % v

            if pc_curve in log.keys():
                data = log[pc_curve]
                depth_index = components.loc[components.UWI == uwi,
                                             'DEPTH_INDEX']
                data[depth_index] = np.copy(components.loc[components.UWI == uwi,
                                                          pc_curve])
            else:
                data = np.empty(len(log[0]))
                data[:] = np.nan
                depth_index = components.loc[components.UWI == uwi,
                                             'DEPTH_INDEX']
                data[depth_index] = np.copy(components.loc[components.UWI == uwi,
                                                          pc_curve])

                log.append_curve(pc_curve, np.copy(data),
                              descr='Principal Component %i from electrofacies' % v)

        if curve_name in log.keys():
            data = log[curve_name]
            depth_index = df.loc[df.UWI == uwi, 'DEPTH_INDEX']
            data[depth_index] = df.loc[df.UWI == uwi, curve_name]
        else:
            data = np.empty(len(log[0]))
            data[:] = np.nan
            depth_index = df.loc[df.UWI == uwi, 'DEPTH_INDEX']

            data[depth_index] = np.copy(df.loc[df.UWI == uwi, curve_name])

            log.append_curve(curve_name, np.copy(data),
                          descr='Electrofacies')

    return logs