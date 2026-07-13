import pandas as pd
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report

# ==========================
# Load Dataset
# ==========================

true = pd.read_csv("True.csv", on_bad_lines="skip", engine="python")
fake = pd.read_csv("Fake.csv", on_bad_lines="skip", engine="python")

true["label"] = 1
fake["label"] = 0

news = pd.concat([true, fake], axis=0)

news = news.drop(["title", "subject", "date"], axis=1)

news = news.sample(frac=1, random_state=42).reset_index(drop=True)

# ==========================
# Text Preprocessing
# ==========================

def wordopt(text):
    text = text.lower()

    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    text = re.sub(r'<.*?>', '', text)

    text = re.sub(r'[^\w\s]', '', text)

    text = re.sub(r'\d', '', text)

    text = re.sub(r'\n', ' ', text)

    return text


news["text"] = news["text"].apply(wordopt)

# ==========================
# Split Dataset
# ==========================

x = news["text"]
y = news["label"]

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.3,
    random_state=42
)

# ==========================
# TF-IDF
# ==========================

vectorization = TfidfVectorizer()

xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)

# ==========================
# Logistic Regression
# ==========================

LR = LogisticRegression()

LR.fit(xv_train, y_train)

pred_lr = LR.predict(xv_test)

print("\nLogistic Regression\n")
print(classification_report(y_test, pred_lr))

# ==========================
# Decision Tree
# ==========================

DTC = DecisionTreeClassifier(random_state=42)

DTC.fit(xv_train, y_train)

pred_dt = DTC.predict(xv_test)

print("\nDecision Tree\n")
print(classification_report(y_test, pred_dt))

# ==========================
# Random Forest
# ==========================

RFC = RandomForestClassifier(random_state=42)

RFC.fit(xv_train, y_train)

pred_rfc = RFC.predict(xv_test)

print("\nRandom Forest\n")
print(classification_report(y_test, pred_rfc))

# ==========================
# Gradient Boosting
# ==========================

GBC = GradientBoostingClassifier(random_state=42)

GBC.fit(xv_train, y_train)

pred_gbc = GBC.predict(xv_test)

print("\nGradient Boosting\n")
print(classification_report(y_test, pred_gbc))

# ==========================
# Prediction Functions
# ==========================

def output_lable(n):
    if n == 0:
        return "Fake News"
    else:
        return "True News"


def manual_testing(news_text):

    testing_news = pd.DataFrame({"text": [news_text]})

    testing_news["text"] = testing_news["text"].apply(wordopt)

    new_x_test = vectorization.transform(testing_news["text"])

    pred_LR = LR.predict(new_x_test)

    pred_DT = DTC.predict(new_x_test)

    pred_GBC = GBC.predict(new_x_test)

    pred_RFC = RFC.predict(new_x_test)

    return {
        "Logistic Regression": output_lable(pred_LR[0]),
        "Decision Tree": output_lable(pred_DT[0]),
        "Gradient Boosting": output_lable(pred_GBC[0]),
        "Random Forest": output_lable(pred_RFC[0])
    }