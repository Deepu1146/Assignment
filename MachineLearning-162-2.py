import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns

def loadPurchaseData(filepath):
    data = pd.read_excel(filepath, sheet_name="Purchase data")
    features = data[["Candies (#)", "Mangoes (Kg)", "Milk Packets (#)"]].values
    output = data["Payment (Rs)"].values.reshape(-1, 1)
    return features, output

def getVectorSpaceDimension(features):
    return features.shape[1]

def getNumberOfVectors(features):
    return features.shape[0]

def calculateMatrixRank(features):
    return np.linalg.matrix_rank(features)

def calculateCostUsingPseudoInverse(features, output):
    pseudoInverse = np.linalg.pinv(features)
    cost = pseudoInverse.dot(output)
    return cost

def classifyCustomers(output, limit=200):
    result = []
    for amount in output:
        if amount > limit:
            result.append("RICH")
        else:
            result.append("POOR")
    return result

def loadStockData(filepath):
    return pd.read_excel(filepath, sheet_name="IRCTC Stock Price")

def meanVarianceUsingNumpy(values):
    return np.mean(values), np.var(values)

def manualMean(values):
    total = 0
    for value in values:
        total += value
    return total / len(values)

def manualVariance(values):
    avg = manualMean(values)
    total = 0
    for value in values:
        total += (value - avg) ** 2
    return total / len(values)

def averageTimeTaken(function, values):
    times = []
    for i in range(10):
        start = time.time()
        function(values)
        end = time.time()
        times.append(end - start)
    return sum(times) / len(times)

def probabilityOfLoss(changePercent):
    losses = list(filter(lambda x: x < 0, changePercent))
    return len(losses) / len(changePercent)

def probabilityProfitOnWednesday(stockdata):
    wednesdayData = stockdata[stockdata["Day"] == "Wednesday"]
    if len(wednesdayData) == 0:
        return 0
    profitDays = wednesdayData[wednesdayData["Chg%"] > 0]
    return len(profitDays) / len(wednesdayData)

def conditionalProbabilityProfitWednesday(stockdata):
    wednesdayData = stockdata[stockdata["Day"] == "Wednesday"]
    if len(wednesdayData) == 0:
        return 0
    profitDays = wednesdayData[wednesdayData["Chg%"] > 0]
    return len(profitDays) / len(wednesdayData)

def plotChangeVsDay(stockdata):
    plt.scatter(stockdata["Day"], stockdata["Chg%"])
    plt.xlabel("Day")
    plt.ylabel("Change Percentage")
    plt.title("Change Percentage vs Day")
    plt.show()

def loadThyroidData(filepath):
    return pd.read_excel(filepath, sheet_name="thyroid0387_UCI")

def analyzeThyroidData(thyroiddata):
    summary = thyroiddata.describe(include="all")
    missingValues = thyroiddata.isnull().sum()
    datatypes = thyroiddata.dtypes
    return summary, missingValues, datatypes

def calculateJaccardAndSmc(vector1, vector2):
    f11 = f10 = f01 = f00 = 0

    for i in range(len(vector1)):
        if vector1[i] == 1 and vector2[i] == 1:
            f11 += 1
        elif vector1[i] == 1 and vector2[i] == 0:
            f10 += 1
        elif vector1[i] == 0 and vector2[i] == 1:
            f01 += 1
        else:
            f00 += 1

    if (f11 + f10 + f01) == 0:
        jaccard = 0
    else:
        jaccard = f11 / (f11 + f10 + f01)

    smc = (f11 + f00) / (f11 + f10 + f01 + f00)

    return jaccard, smc

def calculateCosineSimilarity(vector1, vector2):
    numerator = np.dot(vector1, vector2)
    denominator = np.linalg.norm(vector1) * np.linalg.norm(vector2)
    return numerator / denominator

def plotSimilarityHeatmap(features):
    totalVectors = len(features)
    count = min(20, totalVectors)

    matrix = np.zeros((count, count))

    for i in range(count):
        for j in range(count):
            numerator = np.dot(features[i], features[j])
            denominator = np.linalg.norm(features[i]) * np.linalg.norm(features[j])

            if denominator == 0:
                matrix[i][j] = 0
            else:
                matrix[i][j] = numerator / denominator

    sns.heatmap(matrix, annot=True, cmap="coolwarm")
    plt.title("Cosine Similarity Heatmap")
    plt.show()

def fillMissingValues(dataframe):
    for column in dataframe.columns:
        if dataframe[column].dtype in ["float64", "int64"]:
            dataframe[column].fillna(dataframe[column].mean(), inplace=True)
        else:
            dataframe[column].fillna(dataframe[column].mode()[0], inplace=True)
    return dataframe

def normalizeData(dataframe):
    numericColumns = dataframe.select_dtypes(include=np.number).columns
    for column in numericColumns:
        minValue = dataframe[column].min()
        maxValue = dataframe[column].max()
        dataframe[column] = (dataframe[column] - minValue) / (maxValue - minValue)
    return dataframe

def main():
    filepath = "Lab Session Data (1).xlsx"

    features, output = loadPurchaseData(filepath)
    print("A1 Vector Space Dimension:", getVectorSpaceDimension(features))
    print("A1 Number of Vectors:", getNumberOfVectors(features))
    print("A1 Matrix Rank:", calculateMatrixRank(features))
    print("A1 Item Cost:\n", calculateCostUsingPseudoInverse(features, output))

    print("A2 Customer Classification:", classifyCustomers(output.flatten()))

    stockdata = loadStockData(filepath)
    prices = stockdata.iloc[:, 3].dropna().values

    print("A3 Mean & Variance (NumPy):", meanVarianceUsingNumpy(prices))
    print("A3 Mean Time (Manual):", averageTimeTaken(manualMean, prices))
    print("A3 Variance Time (Manual):", averageTimeTaken(manualVariance, prices))
    print("A3 Probability of Loss:", probabilityOfLoss(stockdata["Chg%"].dropna()))
    print("A3 Probability of Profit on Wednesday:", probabilityProfitOnWednesday(stockdata))
    print("A3 Conditional Probability Profit | Wednesday:",
          conditionalProbabilityProfitWednesday(stockdata))

    plotChangeVsDay(stockdata)

    thyroiddata = loadThyroidData(filepath)
    summary, missing, types = analyzeThyroidData(thyroiddata)
    print("A4 Summary:\n", summary)
    print("A4 Missing Values:\n", missing)
    print("A4 Data Types:\n", types)

    print("A5 Jaccard & SMC:", calculateJaccardAndSmc(features[0], features[1]))
    print("A6 Cosine Similarity:", calculateCosineSimilarity(features[0], features[1]))

    plotSimilarityHeatmap(features)

    thyroiddata = fillMissingValues(thyroiddata)
    thyroiddata = normalizeData(thyroiddata)
    print("A8 & A9 Data Imputation and Normalization Completed")

if __name__ == "__main__":
    main()
