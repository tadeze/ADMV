import fancyimpute as fi



def impute_value(df):
    mice_impute = fi.MICE().complete(df)
    return mice_impute
