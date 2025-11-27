import pandas as pd


def rename_uids(df: pd.DataFrame) -> pd.DataFrame:
    if "fold" in df.index[0]:
        base_uid_list = df.index.str.split('_').map(
            lambda x: '_'.join(x[:-2]) if len(x) > 2 else df.index[0])
        df_cln = df.copy()
        df_cln.index = base_uid_list
        df_cln = df_cln.groupby(level=0).mean()
    else:
        df_cln = df

    return df_cln
