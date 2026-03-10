import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.cells = [
    nbf.v4.new_markdown_cell("# EDA Bone Marrow Transplant"),
    nbf.v4.new_code_cell(
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n"
        "import seaborn as sns\n"
        "df = pd.read_csv('../data/bone_marrow.csv')\n"
        "print(df.shape)\n"
        "df.head()"
    ),
    nbf.v4.new_code_cell(
        "df['survival_status'].value_counts().plot(kind='bar', color=['red','blue'])\n"
        "plt.title('Survival Distribution')\n"
        "plt.show()"
    ),
    nbf.v4.new_code_cell(
        "import numpy as np\n"
        "import matplotlib.pyplot as plt\n"
        "sns.heatmap(df.select_dtypes(include=['number']).corr(), cmap='RdBu_r', center=0)\n"
        "plt.title('Correlation Heatmap')\n"
        "plt.show()"
    ),
]

nbf.write(nb, open('notebooks/eda.ipynb', 'w', encoding='utf-8'))
print('OK!')
