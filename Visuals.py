import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
dataset = pd.read_csv(filepath_or_buffer="DataSource/binary.csv")
plt.scatter(dataset['gre'], dataset['gpa'], c=dataset['rank'])
plt.show()


plot = sns.pairplot(data=dataset)
plot.savefig('visual_1.png', dpi=500)