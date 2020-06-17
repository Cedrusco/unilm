
import pandas as pd
mapping_df = pd.read_csv('mapping.csv')


def get_label(template_id):
    row = mapping_df[mapping_df["template_id"] == template_id] 
    label = row["label"].values[0]
    print("get_label ", label)
    return label

def get_template_id(label):
    row = mapping_df[mapping_df["label"] == label] 
    print(row)
    template_id = row["template_id"].values[0]
    print("get_template_id ", template_id)
    return template_id

def check_if_exists(template_id):
    template_ids = mapping_df["template_id"].values
    exists = template_id in template_ids
    print("check_if_exists ", exists)
    return exists

def add_template_id(template_id):
    print(mapping_df)
    max_label = mapping_df['label'].max()
    new_label = int(max_label) +1
    new_row = [new_label ,template_id]
    mapping_df.loc[len(mapping_df)] = new_row
    mapping_df.to_csv('mapping.csv', header=True, mode = 'w', index=False)
    return new_label

def max_label():
    return mapping_df['label'].max()


if __name__ == "__main__":
    get_template_id(0)