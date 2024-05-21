import regex as re

def contains_target_phrases(sentence):
    # Compile a regular expression pattern to search for the phrases, making it case-insensitive
    pattern = re.compile(r'background liver|background mediastinal blood pool| background liver', re.IGNORECASE)

    # Search the sentence for the pattern
    if pattern.search(sentence):
        return True
    else:
        return False

def template_removal(df):

    petlymph_to_remove = []
    for index, row in df.iterrows():

        #sent = row["Extracted Sentences"]
        sent = row["sentence"]

        if contains_target_phrases(sent):
            petlymph_to_remove.append(row["Petlymph"])

    df = df[~df["Petlymph"].isin(petlymph_to_remove)]
    return df