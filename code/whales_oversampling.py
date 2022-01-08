import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np
import warnings
import os
warnings.simplefilter("ignore", category=DeprecationWarning)

print(os.listdir("oversampling"))
LABELS = 'oversampling/train.csv'

train = pd.read_csv(LABELS)


def oversampling(df, train_file_name, train_adnf_val_file_name):
    im_count = df[df.Id != 'new_whale'].Id.value_counts()
    im_count.name = 'sighting_count'
    df = df.join(im_count, on='Id')
    val_fns = set(df.sample(frac=1)[(df.Id != 'new_whale') & (df.sighting_count > 1)].groupby('Id').first().Image)

    # remove new whale
    df = df[df.Id != 'new_whale']
    print(df.shape)
    print("With max sighting count:")
    print(df.sighting_count.max())

    print("Amount of validation file names: " + str(len(val_fns)))
    df_val = df[df.Image.isin(val_fns)]
    df_train = df[~df.Image.isin(val_fns)]
    df_train_with_val = df

    print(df_val.shape, df_train.shape, df_train_with_val.shape)

    df_train_oversampled = up_sample(15, df_train)
    print("Oversampled shape: " + str(np.shape(df_train_oversampled)))

    df_train_oversampled_with_validation = up_sample(15, df_train_with_val)
    print("Oversampled with validation shape: " + str(np.shape(df_train_oversampled_with_validation)))

    pd.concat((df_train_oversampled, df_val))[['Image', 'Id']].to_csv(train_file_name, index=False)

    # shuffle oversampled train and validation dataframe
    df_train_oversampled_with_validation = df_train_oversampled_with_validation.iloc[np.random.permutation(len(df_train_oversampled_with_validation))]
    df_train_oversampled_with_validation[['Image', 'Id']].to_csv(train_adnf_val_file_name, index=False)


def up_sample(sample_to, df_train):
    res = None

    for grp in df_train.groupby('Id'):
        n = grp[1].shape[0]
        additional_rows = grp[1].sample(0 if sample_to < n else sample_to - n, replace=True)
        rows = pd.concat((grp[1], additional_rows))

        if res is None:
            res = rows
        else:
            res = pd.concat((res, rows))
    return res


train = pd.read_csv(LABELS)
oversampling(train, "oversampling/oversampled_train.csv", "oversampling/oversampled_train_and_val_shuffled.csv")

