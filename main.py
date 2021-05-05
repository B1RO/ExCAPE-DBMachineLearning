# Press the green button in the gutter to run the script.
import numbers
import pathlib
import itertools
import json

import pandas as pd
import matplotlib.pyplot as plt
import os
from mordred import Calculator, descriptors, error
from rdkit import Chem
import numpy as np
from sklearn import datasets


def read_downloaded_dict_as_df(path):
    with os.scandir(path) as it:
        every_nth = itertools.islice(it, None, None, 10)
        json_contents = [json.loads(pathlib.Path(entry.path).read_text()) for entry in every_nth]
        list_of_doc_lists = [content["response"]["docs"] for content in json_contents]
        docs = list(itertools.chain.from_iterable(list_of_doc_lists))
        return pd.DataFrame(docs)


def get_invalid_smiles_indices(mols: list):
    return [i for i, x in enumerate(mols) if x == None]


def calculate_molecular_descriptors(df: pd.DataFrame) -> pd.DataFrame:
    calc = Calculator(descriptors, ignore_3D=True)
    mols = [Chem.MolFromSmiles(smi) for smi in df.SMILES]
    invalid_indices = get_invalid_smiles_indices(mols)
    mols_without_invalid = [mol for index, mol in enumerate(mols) if index not in invalid_indices]
    descriptor_df = calc.pandas(mols_without_invalid)
    return df.drop(df.index[invalid_indices]).join(descriptor_df)


def drop_null_columns(df: pd.DataFrame) -> pd.DataFrame:
    null_counts = df.isnull().sum()
    print("Dropping the following columns with null values : \n{}".format(null_counts[null_counts >= 0.2 * len(df)]))
    withoutCols = df.drop(null_counts[null_counts > len(df) * 0.2].index, axis=1)
    return withoutCols.dropna();


def cleanup_object_columns(df: pd.DataFrame) -> pd.DataFrame:
    errorLabels = ['Timeout', 'DuplicatedDescriptorName', 'Missing', 'Error', 'Missing3DCoordinate', 'MissingValueBase',
                   'MultipleFragments', 'MordredException', 'ABCMeta']
    numDropped = 0;
    for i, colName in enumerate(df.select_dtypes(include=['object'])):
        col = df[colName]
        numericEntries = col.map(lambda x: isinstance(x, numbers.Number))
        errorEntriesByType = [col.map(lambda x: type(x) == errorType) for errorType in
                              [error.Timeout, error.DuplicatedDescriptorName, error.Missing, error.Error,
                               error.Missing3DCoordinate, error.MissingValueBase, error.MultipleFragments,
                               error.MordredException, error.ABCMeta]]
        if sum(errorEntriesByType[errorLabels.index('Missing')]) / len(col) > 0.025:
            df.drop(colName, axis=1, inplace=True)
            numDropped += 1;
        else:
            df[colName] = pd.to_numeric(df[colName], errors='coerce').fillna(method='ffill')
    return df;


def plot_object_columns(df: pd.DataFrame) -> pd.DataFrame:
    colors = ['#b71c1c', '#4A148C', '#D81B60', '#01579B', '#004D40', '#33691E', '#F57F17', '#3E2723']
    errorLabels = ['Timeout', 'DuplicatedDescriptorName', 'Missing', 'Error', 'Missing3DCoordinate', 'MissingValueBase',
                   'MultipleFragments', 'MordredException', 'ABCMeta']
    for i, colName in enumerate(df.select_dtypes(include=['object'])):
        col = df[colName]
        numericEntries = col.map(lambda x: isinstance(x, numbers.Number))
        errorEntriesByType = [col.map(lambda x: type(x) == errorType) for errorType in
                              [error.Timeout, error.DuplicatedDescriptorName, error.Missing, error.Error,
                               error.Missing3DCoordinate, error.MissingValueBase, error.MultipleFragments,
                               error.MordredException, error.ABCMeta]]
        numericValueCount = numericEntries.sum()
        plt.plot(col[numericEntries], label='numeric value', marker='o', linestyle='none')
        for label, color, errorTypeEntries in zip(errorLabels, colors, errorEntriesByType):
            if np.sum(errorTypeEntries) != 0:
                plt.plot((errorTypeEntries[errorTypeEntries == True]).map(lambda x: 0), color=color, label=label,
                         marker='o', linestyle='none')
        plt.figtext(0.5, 0.01, str(numericValueCount) + " / " + str(len(col)), wrap=True, horizontalalignment='center',
                    fontsize=12)
        plt.legend();
        plt.savefig('figs/' + str(i) + ".png");
        plt.close();


def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)


if __name__ == '__main__':
    df = read_downloaded_dict_as_df('./out_09_02_02__2021_05_02')
    descriptors = cleanup_object_columns(drop_null_columns(calculate_molecular_descriptors(df)))
    descriptors['Activity_Flag'] = descriptors['Activity_Flag'] == 'A'
    descriptors_clean = clean_dataset(descriptors.drop(
        ['Original_Entry_ID', 'Entrez_ID', 'Activity_Flag', 'DB', 'Original_Assay_ID', 'Tax_ID', 'Gene_Symbol',
         "Ortholog_Group", 'SMILES', "Ambit_InchiKey"], axis=1))
    descriptors_clean = descriptors_clean.reset_index()
    as_numpy = descriptors_clean.to_numpy()
    datasets.dump_svmlight_file(as_numpy, descriptors['Activity_Flag'].to_numpy(), "svm_out.svmlight")
