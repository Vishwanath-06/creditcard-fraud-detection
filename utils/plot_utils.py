import seaborn as sns
import matplotlib.pyplot as plt

def plot_distribution(df):
    plt.figure(figsize=(6,4))
    sns.countplot(x="Class", data=df)
    plt.title("Fraud vs Non-Fraud Distribution")
    plt.show()
