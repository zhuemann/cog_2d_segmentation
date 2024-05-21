import regex as re

def contains_target_phrases(sentence):
    # Compile a regular expression pattern to search for the phrases, making it case-insensitive
    pattern = re.compile(r'background liver|background mediastinal blood pool|…Background liver', re.IGNORECASE)

    # Search the sentence for the pattern
    if pattern.search(sentence):
        return True
    else:
        return False

def template_removal(df):

    labels_to_remove = []
    for index, row in df.iterrows():

        sent = row["sentence"]

        if contains_target_phrases(sent):
            labels_to_remove.append(row["Label_Name"])

    print(f"Label name removed: {labels_to_remove}")
    df = df[~df["Label_Name"].isin(labels_to_remove)]
    return df