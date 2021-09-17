import os
import re
import argparse
import pandas as pd
from pyathena import connect
from itertools import chain

def extract_code(x):
    codes = []
    output = re.findall(r"(Code:\s*\s*([0-9A-Z:. ]+))", str(x))
    if len(output) > 0:
        for code in output:
            x = code[1].split(",")[0]
            if x not in codes:
                codes.append(x)
        return codes
    else:
        return codes
    
def extract_patient_id(x):
    x = x['reference']
    x = x.split("/")[-1]
    x = x.split("}")[0]
    return x

def extract_description(x):
    description = []
    output = re.findall(r"(Description:.*?(?=(?:,Code)|$))", str(x))
    if len(output) > 0:
        for code in output:
            description.append(code.replace('Description: ',''))
        return description
    else:
        return description
    
def remove_nested_list(x):
    return list(chain(*x))


def process_hl_data(base_dir):
    df_document_reference = pd.read_json("{}/DocumentReference.ndjson".format(base_dir), lines=True)
    df_document_reference['code_value'] = df_document_reference['extension'].apply(extract_code)
    df_document_reference['patient_id'] = df_document_reference['subject'].apply(extract_patient_id)
    df_document_reference['code_description'] = df_document_reference['extension'].apply(extract_description)
    df_document_reference = df_document_reference[['patient_id', 'code_value', 'code_description']]
    df_document_reference = df_document_reference.groupby(['patient_id']).agg(lambda x: tuple(x)).applymap(list).reset_index()
    df_document_reference['code_value'] = df_document_reference['code_value'].apply(remove_nested_list)
    df_document_reference['code_description'] = df_document_reference['code_description'].apply(remove_nested_list)
    return df_document_reference    