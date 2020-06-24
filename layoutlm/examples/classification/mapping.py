import pandas as pd
mapping_df = pd.read_csv('mapping.csv')

# This file contains util methods pertaining to reading and writing to mapping.csv


# Lookup a template ID in mapping.csv and return its corresponding label
# Expects one argument, template_id - template ID to look up in mapping.csv
def get_label(template_id):
    row = mapping_df[mapping_df["template_id"] == template_id] 
    label = row["label"].values[0]
    print("get_label ", label)
    return label


# Lookup a label in mapping.csv and return its corresponding template ID
# Expects one argument, label - label to look up in mapping.csv
def get_template_id(label):
    row = mapping_df[mapping_df["label"] == label] 
    print(row)
    template_id = row["template_id"].values[0]
    print("get_template_id ", template_id)
    return template_id


# Lookup a template ID in mapping.csv and return whether or not it exists (boolean)
# Expects one argument, template_id - template ID to look up in mapping.csv
def check_if_exists(template_id):
    template_ids = mapping_df["template_id"].values
    exists = template_id in template_ids
    print("check_if_exists ", exists)
    return exists


# Write to mapping.csv and add a new template_id, mapping it to a new label
# Expects one argument, template_id - template ID to add in mapping.csv
def add_template_id(template_id):
    print(mapping_df)
    label_max = mapping_df['label'].max()
    new_label = int(label_max) + 1
    new_row = [new_label, template_id]
    mapping_df.loc[len(mapping_df)] = new_row
    mapping_df.to_csv('mapping.csv', header=True, mode='w', index=False)
    return new_label


# Return the last/greatest label within mapping.csv, used to determine label for a new template ID
def max_label():
    return mapping_df['label'].max()


if __name__ == "__main__":
    get_template_id(0)
