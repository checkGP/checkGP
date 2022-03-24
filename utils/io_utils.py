import pandas as pd
import numpy as np


def read_mauna_loa(file_path, date_split_predict=2003, normalized_data=False):
    # Load the data
    # Load the data from the Scripps CO2 program website.
    co2_df = pd.read_csv(
        # Source: https://scrippsco2.ucsd.edu/assets/data/atmospheric/stations/in_situ_co2/monthly/monthly_in_situ_co2_mlo.csv
        # 'maunaloa.csv',
        file_path,
        header=7,  # Data starts here
        skiprows=[55, 56],  # Headers consist of multiple rows
        usecols=[3, 4],  # Only keep the 'Date' and 'CO2' columns
        na_values='-99.99',  # NaNs are denoted as '-99.99'
        names=["Date", "CO2"],
        dtype=np.float64
    )

    # Drop missing values
    co2_df.dropna(inplace=True)
    # Remove whitespace from column names
    co2_df.rename(columns=lambda x: x.strip(), inplace=True)

    df_observed = co2_df[co2_df.Date < date_split_predict]
    print('{} measurements in the observed set'.format(len(df_observed)))
    df_predict = co2_df[co2_df.Date >= date_split_predict]
    print('{} measurements in the test set'.format(len(df_predict)))
    y = df_observed.CO2.values.reshape(-1, 1)
    x = df_observed.Date.values.reshape(-1, 1)
    ytest = df_predict.CO2.values.reshape(-1, 1)
    xtest = df_predict.Date.values.reshape(-1, 1)
    if normalized_data:
        ymu = y.mean()
        ystd = np.std(y)
        maxX = np.max(x)
        y -= ymu
        y /= ystd
        ytest -= ymu
        ytest /= ystd
        x /= maxX
        xtest /= maxX
    else:
        ymu = 0
        ystd = 1
        maxX = 1
    scale_params = {'ymu': ymu, 'ystd': ystd, 'maxX': maxX}

    return y.ravel(), x.ravel(), ytest.ravel(), xtest.ravel(), scale_params