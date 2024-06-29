# 导入相关的包
import warnings
warnings.filterwarnings('ignore')
import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
import pandas as pd
import numpy as np

# 数据集的路径
data_path = "./datasets/5f9ae242cae5285cd734b91e-momodel/sms_pub.csv"

# 读取数据
sms = pd.read_csv(data_path, encoding='utf-8')

# ---------------------------------------------------

def read_stopwords(stopwords_path):
    """
    读取停用词库
    :param stopwords_path: 停用词库的路径
    :return: 停用词列表
    """
    with open(stopwords_path, 'r', encoding='utf-8') as f:
        stopwords = f.read()
    stopwords = stopwords.splitlines()
    return stopwords

# 停用词库路径
stopwords_path = r'scu_stopwords.txt'
# 读取停用词
stopwords = read_stopwords(stopwords_path)

# 构建训练集和测试集
from sklearn.model_selection import train_test_split
X = np.array(sms['msg_new'])
y = np.array(sms['label'])
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.1)

# ----------------- 导入相关的库 -----------------
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn import preprocessing

# pipline_list用于传给Pipline作为参数
pipeline_list = [
    # --------------------------- 需要完成的代码 ----------------------
    # 以下代码仅供参考
    ('tv', CountVectorizer(token_pattern=r"(?u)\b\w+\b", stop_words=stopwords)),
    ('classifier', MultinomialNB()),
    # 以上代码仅供参考
    # ---------------------------------------------------------------------
]

# 搭建 pipeline
pipeline = Pipeline(pipeline_list)

# 训练 pipeline
pipeline.fit(X_train, y_train)

# 对测试集的数据集进行预测
y_pred = pipeline.predict(X_test)

# 在测试集上进行评估
from sklearn import metrics
print("在测试集上的混淆矩阵：")
print(metrics.confusion_matrix(y_test, y_pred))
print("在测试集上的分类结果报告：")
print(metrics.classification_report(y_test, y_pred))
print("在测试集上的 f1-score：")
print(metrics.f1_score(y_test, y_pred))

# 在所有的样本上训练一次，充分利用已有的数据，提高模型的泛化能力
X = np.array(sms['msg_new'])
y = np.array(sms['label'])
pipeline.fit(X, y)

# 保存训练的模型，请将模型保存在 results 目录下
from joblib import dump  # 注意 sklearn.externals 已经弃用，使用 joblib 直接导入
pipeline_path = 'results/pipeline.model'
dump(pipeline, pipeline_path)