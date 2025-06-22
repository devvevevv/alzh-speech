import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

EVAL_CONTROL_PATH = r"..\data\processed\linguistic_features\control\eval_control.csv"
EVAL_AD_PATH = r"..\data\processed\linguistic_features\dementia\eval_dementia.csv"
FLUCALC_CONTROL_PATH = r"..\data\processed\linguistic_features\control\flucalc_control.csv"
FLUCALC_AD_PATH = r"..\data\processed\linguistic_features\dementia\flucalc_dementia.csv"
OUTPUT_PATH = r"..\data\features.csv"

EVAL_SELECTED_FEATURES = [
    "Age", "Sex", "Group",
    "Duration_(sec)", "MLU_Utts", "MLU_Morphemes",
    "FREQ_TTR", "Words_Min", "Verbs_Utt",
    "%_Word_Errors", "Utt_Errors", "density",
    "%_Nouns", "%_Plurals", "%_Verbs",
    "%_Aux", "%_Mod", "%_3S",
    "%_13S", "%_PAST", "%_PASTP",
    "%_PRESP", "%_prep", "%_adj",
    "%_adv", "%_conj", "%_det",
    "%_pro", "noun_verb", "retracing", "repetition"
]

FLUCALC_SELECTED_FEATURES = [
    'mor_Utts', 'mor_syllables', 'syllables_min', '%_Prolongation',
    'Mean_RU', '%_Phonological_fragment', '%_Phrase_repetitions',
    '%_Word_revisions', '%_Phrase_revisions', '%_Pauses', '%_Filled_pauses',
    '%_TD', 'SLD_Ratio', 'Content_words_ratio', 'Function_words_ratio'
]

#for IPSYN features
"""
IPSYN_SELECTED_FEATURES = [
    "N", "V", "Q", "S"
]
"""

eval_control = pd.read_csv(EVAL_CONTROL_PATH)[EVAL_SELECTED_FEATURES]
eval_dementia = pd.read_csv(EVAL_AD_PATH)[EVAL_SELECTED_FEATURES]

flucalc_control = pd.read_csv(FLUCALC_CONTROL_PATH)[FLUCALC_SELECTED_FEATURES]
flucalc_dementia = pd.read_csv(FLUCALC_AD_PATH)[FLUCALC_SELECTED_FEATURES]

control_combined = pd.concat([eval_control, flucalc_control], axis=1)
dementia_combined = pd.concat([eval_dementia, flucalc_dementia], axis=1)

control_combined["Label"] = 0
dementia_combined["Label"] = 1

df = pd.concat([control_combined, dementia_combined], axis=0).reset_index(drop=True)

#Clean 'Age' and convert to float
df["Age"] = (
    df["Age"]
    .astype(str)
    .str.replace(";", "", regex=False)
    .str.replace(":", "", regex=False)
    .str.replace(",", "", regex=False)
    .str.strip()
)
df["Age"] = pd.to_numeric(df["Age"], errors="coerce")

df = df.drop(columns = ["Group"]).dropna()

le_sex = LabelEncoder()
if "Sex" in df.columns:
    df["Sex"] = le_sex.fit_transform(df["Sex"])

X = df.drop(columns = ["Label"])
y = df["Label"]

feature_columns = X.columns
scaler = StandardScaler()
X = scaler.fit_transform(X)

df_cleaned = pd.DataFrame(X, columns=feature_columns)
df_cleaned["Label"] = y.values
df_cleaned.to_csv(OUTPUT_PATH, index=False)

print("Preprocessing complete.")
print(f"Saved cleaned dataset to: {OUTPUT_PATH}")
