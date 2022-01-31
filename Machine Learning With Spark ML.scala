// Databricks notebook source
// MAGIC %md
// MAGIC # Machine Learning With Spark ML
// MAGIC 
// MAGIC Levent Güner <leventg@kth.se>
// MAGIC 
// MAGIC I. Erdem Demir <iedemir@kth.se>
// MAGIC 
// MAGIC In this lab assignment, you will complete a project by going through the following steps:
// MAGIC 1. Get the data.
// MAGIC 2. Discover the data to gain insights.
// MAGIC 3. Prepare the data for Machine Learning algorithms.
// MAGIC 4. Select a model and train it.
// MAGIC 5. Fine-tune your model.
// MAGIC 6. Present your solution.
// MAGIC 
// MAGIC As a dataset, we use the California Housing Prices dataset from the StatLib repository. This dataset was based on data from the 1990 California census. The dataset has the following columns
// MAGIC 1. `longitude`: a measure of how far west a house is (a higher value is farther west)
// MAGIC 2. `latitude`: a measure of how far north a house is (a higher value is farther north)
// MAGIC 3. `housing_,median_age`: median age of a house within a block (a lower number is a newer building)
// MAGIC 4. `total_rooms`: total number of rooms within a block
// MAGIC 5. `total_bedrooms`: total number of bedrooms within a block
// MAGIC 6. `population`: total number of people residing within a block
// MAGIC 7. `households`: total number of households, a group of people residing within a home unit, for a block
// MAGIC 8. `median_income`: median income for households within a block of houses
// MAGIC 9. `median_house_value`: median house value for households within a block
// MAGIC 10. `ocean_proximity`: location of the house w.r.t ocean/sea
// MAGIC 
// MAGIC ---
// MAGIC # 1. Get the data
// MAGIC Let's start the lab by loading the dataset. The can find the dataset at `data/housing.csv`. To infer column types automatically, when you are reading the file, you need to set `inferSchema` to true. Moreover enable the `header` option to read the columns' name from the file.

// COMMAND ----------

import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Matrix
import org.apache.spark.ml.stat.Correlation
import org.apache.spark.sql.Row
import org.apache.spark.ml.feature.Imputer
import org.apache.spark.ml.feature.{VectorAssembler, StandardScaler}
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.OneHotEncoder
import org.apache.spark.ml.{Pipeline, PipelineModel, PipelineStage}
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.evaluation.{RegressionEvaluator,BinaryClassificationEvaluator,MulticlassClassificationEvaluator}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.ml.regression.DecisionTreeRegressor
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.regression.RandomForestRegressor
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.regression.GBTRegressor
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.tuning.ParamGridBuilder
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.tuning.CrossValidator
import scala.collection.mutable.ArrayBuffer
import org.apache.spark.sql.DataFrame

// COMMAND ----------

val housing = spark.read.format("csv")
    .option("header", "true")
    .option("inferschema", "true")
    .load("dbfs:/FileStore/shared_uploads/leventg@kth.se/data/housing.csv")

// COMMAND ----------

// MAGIC %md
// MAGIC ---
// MAGIC # 2. Discover the data to gain insights
// MAGIC Now it is time to take a look at the data. In this step we are going to take a look at the data a few different ways:
// MAGIC * See the schema and dimension of the dataset
// MAGIC * Look at the data itself
// MAGIC * Statistical summary of the attributes
// MAGIC * Breakdown of the data by the categorical attribute variable
// MAGIC * Find the correlation among different attributes
// MAGIC * Make new attributes by combining existing attributes

// COMMAND ----------

// MAGIC %md
// MAGIC ## 2.1. Schema and dimension
// MAGIC Print the schema of the dataset

// COMMAND ----------

housing.printSchema()

// COMMAND ----------

// MAGIC %md
// MAGIC Print the number of records in the dataset.

// COMMAND ----------

housing.count

// COMMAND ----------

// MAGIC %md
// MAGIC ## 2.2. Look at the data
// MAGIC Print the first five records of the dataset.

// COMMAND ----------

housing.show(5, false)

// COMMAND ----------

// MAGIC %md
// MAGIC Print the number of records with population more than 10000.

// COMMAND ----------

housing.filter($"population" > 10000).count

// COMMAND ----------

// MAGIC %md
// MAGIC ## 2.3. Statistical summary
// MAGIC Print a summary of the table statistics for the attributes `housing_median_age`, `total_rooms`, `median_house_value`, and `population`. You can use the `describe` command.

// COMMAND ----------

housing.describe("housing_median_age", "total_rooms", "median_house_value", "population").show()

// COMMAND ----------

// MAGIC %md
// MAGIC Print the maximum age (`housing_median_age`), the minimum number of rooms (`total_rooms`), and the average of house values (`median_house_value`).

// COMMAND ----------

housing.agg(max("housing_median_age")).show()
housing.agg(min("total_rooms")).show()
housing.agg(avg("median_house_value")).show()

// COMMAND ----------

// MAGIC %md
// MAGIC ## 2.4. Breakdown the data by categorical data
// MAGIC Print the number of houses in different areas (`ocean_proximity`), and sort them in descending order.

// COMMAND ----------

housing.groupBy("ocean_proximity")
    .count()
    .sort(desc("count"))
    .show()

// COMMAND ----------

// MAGIC %md
// MAGIC Print the average value of the houses (`median_house_value`) in different areas (`ocean_proximity`), and call the new column `avg_value` when print it.

// COMMAND ----------

housing.groupBy("ocean_proximity")
    .agg(avg("median_house_value"))
    .withColumnRenamed("avg(median_house_value)",  "avg_value")
    .show()

// COMMAND ----------

// MAGIC %md
// MAGIC Rewrite the above question in SQL.

// COMMAND ----------

housing.createOrReplaceTempView("df")
spark.sql("SELECT ocean_proximity, avg(median_house_value) AS avg_value FROM df GROUP BY ocean_proximity").show()

// COMMAND ----------

// MAGIC %md
// MAGIC ## 2.5. Correlation among attributes
// MAGIC Print the correlation among the attributes `housing_median_age`, `total_rooms`, `median_house_value`, and `population`. To do so, first you need to put these attributes into one vector. Then, compute the standard correlation coefficient (Pearson) between every pair of attributes in this new vector. To make a vector of these attributes, you can use the `VectorAssembler` Transformer.

// COMMAND ----------

val va = new VectorAssembler().setInputCols(Array("housing_median_age","total_rooms","median_house_value","population")).setOutputCol("features")

val housingAttrs = va.transform(housing)

housingAttrs.show(5)

// COMMAND ----------

val Row(coeff: Matrix) = Correlation.corr(housingAttrs, "features").head

println(s"The standard correlation coefficient:\n ${coeff}")

// COMMAND ----------

// MAGIC %md
// MAGIC ## 2.6. Combine and make new attributes
// MAGIC Now, let's try out various attribute combinations. In the given dataset, the total number of rooms in a block is not very useful, if we don't know how many households there are. What we really want is the number of rooms per household. Similarly, the total number of bedrooms by itself is not very useful, and we want to compare it to the number of rooms. And the population per household seems like also an interesting attribute combination to look at. To do so, add the three new columns to the dataset as below. We will call the new dataset the `housingExtra`.
// MAGIC ```
// MAGIC rooms_per_household = total_rooms / households
// MAGIC bedrooms_per_room = total_bedrooms / total_rooms
// MAGIC population_per_household = population / households
// MAGIC ```

// COMMAND ----------

val housingCol1 = housing.withColumn("rooms_per_household", expr("total_rooms/households"))
val housingCol2 = housingCol1.withColumn("bedrooms_per_room", expr("total_bedrooms/total_rooms"))
val housingExtra = housingCol2.withColumn("population_per_household", expr("population/households"))

housingExtra.select("rooms_per_household", "bedrooms_per_room", "population_per_household").show(5)

// COMMAND ----------

// MAGIC %md
// MAGIC ---
// MAGIC ## 3. Prepare the data for Machine Learning algorithms
// MAGIC Before going through the Machine Learning steps, let's first rename the label column from `median_house_value` to `label`.

// COMMAND ----------

val renamedHousing = housingExtra.withColumnRenamed("median_house_value","label")
renamedHousing.show()

// COMMAND ----------

// MAGIC %md
// MAGIC Now, we want to separate the numerical attributes from the categorical attribute (`ocean_proximity`) and keep their column names in two different lists. Moreover, sice we don't want to apply the same transformations to the predictors (features) and the label, we should remove the label attribute from the list of predictors. 

// COMMAND ----------


val colLabel = "label"

val colCat = "ocean_proximity"

val colNum = renamedHousing.columns.filter(_ != colLabel).filter(_ != colCat)

// COMMAND ----------

// MAGIC %md
// MAGIC ## 3.1. Prepare continuse attributes
// MAGIC ### Data cleaning
// MAGIC Most Machine Learning algorithms cannot work with missing features, so we should take care of them. As a first step, let's find the columns with missing values in the numerical attributes. To do so, we can print the number of missing values of each continues attributes, listed in `colNum`.

// COMMAND ----------

printf("Missing values for columns\n")
printf("-"*30)
for (c <- colNum) {
    val naCount = renamedHousing.filter(renamedHousing(c).isNull || renamedHousing(c).isNaN).count() 
    printf("\n"+c + " - " + naCount)
}

// COMMAND ----------

// MAGIC %md
// MAGIC As we observerd above, the `total_bedrooms` and `bedrooms_per_room` attributes have some missing values. One way to take care of missing values is to use the `Imputer` Transformer, which completes missing values in a dataset, either using the mean or the median of the columns in which the missing values are located. To use it, you need to create an `Imputer` instance, specifying that you want to replace each attribute's missing values with the "median" of that attribute.

// COMMAND ----------

val imputer = new Imputer().setStrategy("median")
                           .setInputCols(Array("total_bedrooms", "bedrooms_per_room"))
                           .setOutputCols(Array("total_bedrooms", "bedrooms_per_room")) 
val imputedHousing = imputer.fit(renamedHousing).transform(renamedHousing)

imputedHousing.select("total_bedrooms", "bedrooms_per_room").show(5)

// COMMAND ----------

// MAGIC %md
// MAGIC ### Scaling
// MAGIC One of the most important transformations you need to apply to your data is feature scaling. With few exceptions, Machine Learning algorithms don't perform well when the input numerical attributes have very different scales. This is the case for the housing data: the total number of rooms ranges from about 6 to 39,320, while the median incomes only range from 0 to 15. Note that scaling the label attribues is generally not required.
// MAGIC 
// MAGIC One way to get all attributes to have the same scale is to use standardization. In standardization, for each value, first it subtracts the mean value (so standardized values always have a zero mean), and then it divides by the variance so that the resulting distribution has unit variance. To do this, we can use the `StandardScaler` Estimator. To use `StandardScaler`, again we need to convert all the numerical attributes into a big vectore of features using `VectorAssembler`, and then call `StandardScaler` on that vactor.

// COMMAND ----------

val va = new VectorAssembler()
    .setInputCols(colNum)
    .setOutputCol("features")
val featuredHousing = va.transform(imputedHousing)

val scaler = new StandardScaler()
    .setInputCol("features")
    .setOutputCol("scaled")
val scaledHousing = scaler.fit(featuredHousing).transform(featuredHousing)

scaledHousing.show(5)

// COMMAND ----------

// MAGIC %md
// MAGIC ## 3.2. Prepare categorical attributes
// MAGIC After imputing and scaling the continuse attributes, we should take care of the categorical attributes. Let's first print the number of distict values of the categirical attribute `ocean_proximity`.

// COMMAND ----------

renamedHousing.select(colCat).groupBy(colCat)
    .count()
    .show()

// COMMAND ----------

// MAGIC %md
// MAGIC ### String indexer
// MAGIC Most Machine Learning algorithms prefer to work with numbers. So let's convert the categorical attribute `ocean_proximity` to numbers. To do so, we can use the `StringIndexer` that encodes a string column of labels to a column of label indices. The indices are in [0, numLabels), ordered by label frequencies, so the most frequent label gets index 0.

// COMMAND ----------

val indexer = new StringIndexer()
    .setInputCol("ocean_proximity")
    .setOutputCol("ocean_proximity_idxs")
val idxHousing = indexer.fit(renamedHousing).transform(renamedHousing)

idxHousing.show(5)

// COMMAND ----------

// MAGIC %md
// MAGIC Now we can use this numerical data in any Machine Learning algorithm. You can look at the mapping that this encoder has learned using the `labels` method: "<1H OCEAN" is mapped to 0, "INLAND" is mapped to 1, etc.

// COMMAND ----------

indexer.fit(renamedHousing).labels

// COMMAND ----------

// MAGIC %md
// MAGIC ### One-hot encoding
// MAGIC Now, convert the label indices built in the last step into one-hot vectors. To do this, you can take advantage of the `OneHotEncoderEstimator` Estimator.

// COMMAND ----------

val encoder = new OneHotEncoder()
    .setInputCols(Array("ocean_proximity_idxs"))
    .setOutputCols(Array("ocean_proximity_onehot"))
val ohHousing = encoder.fit(idxHousing).transform(idxHousing)

ohHousing.show(5)

// COMMAND ----------

// MAGIC %md
// MAGIC ---
// MAGIC # 4. Pipeline
// MAGIC As you can see, there are many data transformation steps that need to be executed in the right order. For example, you called the `Imputer`, `VectorAssembler`, and `StandardScaler` from left to right. However, we can use the `Pipeline` class to define a sequence of Transformers/Estimators, and run them in order. A `Pipeline` is an `Estimator`, thus, after a Pipeline's `fit()` method runs, it produces a `PipelineModel`, which is a `Transformer`.
// MAGIC 
// MAGIC Now, let's create a pipeline called `numPipeline` to call the numerical transformers you built above (`imputer`, `va`, and `scaler`) in the right order from left to right, as well as a pipeline called `catPipeline` to call the categorical transformers (`indexer` and `encoder`). Then, put these two pipelines `numPipeline` and `catPipeline` into one pipeline.

// COMMAND ----------

val numPipeline = new Pipeline()
    .setStages(Array(imputer, va, scaler))
val catPipeline = new Pipeline()
    .setStages(Array(indexer, encoder))
val pipeline = new Pipeline().setStages(Array(numPipeline, catPipeline))
val newHousing = pipeline.fit(renamedHousing).transform(renamedHousing)

newHousing.show(5)

// COMMAND ----------

// MAGIC %md
// MAGIC Now, use `VectorAssembler` to put all attributes of the final dataset `newHousing` into a big vector, and call the new column `features`.

// COMMAND ----------

val lastHousing = newHousing.drop("features")

val va2 = new VectorAssembler()
    .setInputCols(Array("scaled", "ocean_proximity_onehot"))
    .setOutputCol("features")

val dataset = va2
    .transform(lastHousing)
    .select("features", "label")

dataset.show(1, false)

// COMMAND ----------

// MAGIC %md
// MAGIC ---
// MAGIC # 5. Make a model
// MAGIC Here we going to make four different regression models:
// MAGIC * Linear regression model
// MAGIC * Decission tree regression
// MAGIC * Random forest regression
// MAGIC * Gradient-booster forest regression
// MAGIC 
// MAGIC But, before giving the data to train a Machine Learning model, let's first split the data into training dataset (`trainSet`) with 80% of the whole data, and test dataset (`testSet`) with 20% of it.

// COMMAND ----------

val Array(trainSet, testSet) = dataset.randomSplit(Array(0.8, 0.2))

trainSet.show(5)
testSet.show(5)

// COMMAND ----------

// MAGIC %md
// MAGIC ## 5.1. Linear regression model
// MAGIC Now, train a Linear Regression model using the `LinearRegression` class. Then, print the coefficients and intercept of the model, as well as the summary of the model over the training set by calling the `summary` method.

// COMMAND ----------

// train the model
val lr = new LinearRegression()
    .setFeaturesCol("features")
    .setLabelCol("label")
    .setSolver("normal")
    .setMaxIter(10)
val lrModel = lr.fit(trainSet)
val trainingSummary = lrModel.summary

println(s"Coefficients: ${lrModel.coefficients} Intercept: ${lrModel.intercept}")
println(s"RMSE: ${trainingSummary.rootMeanSquaredError}")

// COMMAND ----------

// MAGIC %md
// MAGIC Now, use `RegressionEvaluator` to measure the root-mean-square-erroe (RMSE) of the model on the test dataset.

// COMMAND ----------

// make predictions on the test data
val predictions = lrModel.transform(testSet)
predictions.select("prediction", "label", "features").show(5)

// select (prediction, true label) and compute test error.
val evaluator = new RegressionEvaluator()
    .setMetricName("rmse")
    .setPredictionCol("prediction")
    .setLabelCol("label")
val rmse = evaluator.evaluate(predictions)
println(s"Root Mean Squared Error (RMSE) on test data = $rmse")

// COMMAND ----------

// MAGIC %md
// MAGIC ## 5.2. Decision tree regression
// MAGIC Repeat what you have done on Regression Model to build a Decision Tree model. Use the `DecisionTreeRegressor` to make a model and then measure its RMSE on the test dataset.

// COMMAND ----------

val dt = new DecisionTreeRegressor()
    .setFeaturesCol("features")
    .setLabelCol("label")

// train the model
val dtModel = dt.fit(trainSet)

// make predictions on the test data
val predictions = dtModel.transform(testSet)
predictions.select("prediction", "label", "features").show(5)

// select (prediction, true label) and compute test error
val evaluator = new RegressionEvaluator()
    .setMetricName("rmse")
    .setPredictionCol("prediction")
    .setLabelCol("label")
val rmse = evaluator.evaluate(predictions)
println(s"Root Mean Squared Error (RMSE) on test data = $rmse")

// COMMAND ----------

// MAGIC %md
// MAGIC ## 5.3. Random forest regression
// MAGIC Let's try the test error on a Random Forest Model. Youcan use the `RandomForestRegressor` to make a Random Forest model.

// COMMAND ----------

val rf = new RandomForestRegressor()
    .setNumTrees(25)
    .setMaxDepth(10)
    .setFeatureSubsetStrategy("auto")
    .setLabelCol("label")
    .setFeaturesCol("features")

// train the model
val rfModel = rf.fit(trainSet)

// make predictions on the test data
val predictions = rfModel.transform(testSet)
predictions.select("prediction", "label", "features").show(5)

// select (prediction, true label) and compute test error
val evaluator = new RegressionEvaluator()
    .setMetricName("rmse")
    .setPredictionCol("prediction")
    .setLabelCol("label")
val rmse = evaluator.evaluate(predictions)
println(s"Root Mean Squared Error (RMSE) on test data = $rmse")

// COMMAND ----------

// MAGIC %md
// MAGIC ## 5.4. Gradient-boosted tree regression
// MAGIC Fianlly, we want to build a Gradient-boosted Tree Regression model and test the RMSE of the test data. Use the `GBTRegressor` to build the model.

// COMMAND ----------

val gb = new GBTRegressor()
    .setMaxIter(10)
    .setLabelCol("label")
    .setFeaturesCol("features")

// train the model
val gbModel = gb.fit(trainSet)

// make predictions on the test data
val predictions = gbModel.transform(testSet)
predictions.select("prediction", "label", "features").show(5)

// select (prediction, true label) and compute test error
val evaluator = new RegressionEvaluator()
    .setMetricName("rmse")
    .setPredictionCol("prediction")
    .setLabelCol("label")
val rmse = evaluator.evaluate(predictions)
println(s"Root Mean Squared Error (RMSE) on test data = $rmse")

// COMMAND ----------

// MAGIC %md
// MAGIC ---
// MAGIC # 6. Hyperparameter tuning
// MAGIC An important task in Machie Learning is model selection, or using data to find the best model or parameters for a given task. This is also called tuning. Tuning may be done for individual Estimators such as LinearRegression, or for entire Pipelines which include multiple algorithms, featurization, and other steps. Users can tune an entire Pipeline at once, rather than tuning each element in the Pipeline separately. MLlib supports model selection tools, such as `CrossValidator`. These tools require the following items:
// MAGIC * Estimator: algorithm or Pipeline to tune (`setEstimator`)
// MAGIC * Set of ParamMaps: parameters to choose from, sometimes called a "parameter grid" to search over (`setEstimatorParamMaps`)
// MAGIC * Evaluator: metric to measure how well a fitted Model does on held-out test data (`setEvaluator`)
// MAGIC 
// MAGIC `CrossValidator` begins by splitting the dataset into a set of folds, which are used as separate training and test datasets. For example with `k=3` folds, `CrossValidator` will generate 3 (training, test) dataset pairs, each of which uses 2/3 of the data for training and 1/3 for testing. To evaluate a particular `ParamMap`, `CrossValidator` computes the average evaluation metric for the 3 Models produced by fitting the Estimator on the 3 different (training, test) dataset pairs. After identifying the best `ParamMap`, `CrossValidator` finally re-fits the Estimator using the best ParamMap and the entire dataset.
// MAGIC 
// MAGIC Below, use the `CrossValidator` to select the best Random Forest model. To do so, you need to define a grid of parameters. Let's say we want to do the search among the different number of trees (1, 5, and 10), and different tree depth (5, 10, and 15).

// COMMAND ----------

val paramGrid = new ParamGridBuilder()
    .addGrid(rfModel.maxDepth, Array(1, 5, 10))
    .addGrid(rfModel.numTrees, Array(5, 10, 15))
    .build()

val evaluator = new RegressionEvaluator().setLabelCol("label").setPredictionCol("prediction").setMetricName("rmse")

val newPipeline = new Pipeline().setStages(Array(rfModel)) 

val cv = new CrossValidator()
    .setEstimator(newPipeline)
    .setEstimatorParamMaps(paramGrid)
    .setEvaluator(evaluator)
    .setNumFolds(3)
val cvModel = cv.fit(trainSet)

val predictions = cvModel.transform(testSet)
predictions.select("prediction", "label", "features").show(5)

val rmse = evaluator.evaluate(predictions)
println(s"Root Mean Squared Error (RMSE) on test data = $rmse")

// COMMAND ----------

// MAGIC %md
// MAGIC ---
// MAGIC # 7. An End-to-End Classification Test
// MAGIC As the last step, you are given a dataset called `data/ccdefault.csv`. The dataset represents default of credit card clients. It has 30,000 cases and 24 different attributes. More details about the dataset is available at `data/ccdefault.txt`. In this task you should make three models, compare their results and conclude the ideal solution. Here are the suggested steps:
// MAGIC 1. Load the data.
// MAGIC 2. Carry out some exploratory analyses (e.g., how various features and the target variable are distributed).
// MAGIC 3. Train a model to predict the target variable (risk of `default`).
// MAGIC   - Employ three different models (logistic regression, decision tree, and random forest).
// MAGIC   - Compare the models' performances (e.g., AUC).
// MAGIC   - Defend your choice of best model (e.g., what are the strength and weaknesses of each of these models?).
// MAGIC 4. What more would you do with this data? Anything to help you devise a better solution?

// COMMAND ----------

// MAGIC %md
// MAGIC #### Open Data

// COMMAND ----------

val ccdata = spark.read.format("csv")
    .option("header", "true")
    .option("inferschema", "true")
    .load("dbfs:/FileStore/shared_uploads/leventg@kth.se/data/ccdefault.csv")

// COMMAND ----------

ccdata.printSchema()

// COMMAND ----------

ccdata.count

// COMMAND ----------

ccdata.show(5, false)

// COMMAND ----------

val droppedcc = ccdata.drop("ID")


// COMMAND ----------

ccdata.describe().show()

// COMMAND ----------

// MAGIC %md
// MAGIC #### Separate Columns

// COMMAND ----------

val colCat = Array("SEX", "EDUCATION", "MARRIAGE", "PAY_0", "PAY_2",
                               "PAY_3", "PAY_4", "PAY_5", "PAY_6")

val colNum = Array("LIMIT_BAL","AGE","BILL_AMT1", "BILL_AMT2",
                "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
                "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6")
val colLabel = "label"

// COMMAND ----------

for (c <- colCat) {
    ccdata.groupBy(c)
        .count()
        .sort(desc("count"))
        .show()
}

// COMMAND ----------

// MAGIC %md
// MAGIC #### Describe

// COMMAND ----------

ccdata.describe("LIMIT_BAL","AGE","BILL_AMT1", "BILL_AMT2",
                "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
                "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6").show()

// COMMAND ----------

// MAGIC %md
// MAGIC #### Set Label Column

// COMMAND ----------

val renamedCcdata = ccdata.withColumnRenamed("DEFAULT","label")
renamedCcdata.show()

// COMMAND ----------

// MAGIC %md
// MAGIC #### Check Missing Values

// COMMAND ----------

printf("Missing values for columns\n")
printf("-"*30)
for (c <- colNum) {
    val naCount = renamedCcdata.filter(renamedCcdata(c).isNull || renamedCcdata(c).isNaN).count() 
    printf("\n"+c + " - " + naCount)
}

// COMMAND ----------

printf("Missing values for columns\n")
printf("-"*30)
for (c <- colCat) {
    val naCount = renamedCcdata.filter(renamedCcdata(c).isNull || renamedCcdata(c).isNaN).count() 
    printf("\n"+c + " - " + naCount)
}

// COMMAND ----------

// MAGIC %md
// MAGIC #### Process Numeric Data

// COMMAND ----------

val va = new VectorAssembler()
    .setInputCols(colNum)
    .setOutputCol("features")
val featuredCcdata = va.transform(renamedCcdata)

val scaler = new StandardScaler()
    .setInputCol("features")
    .setOutputCol("scaled")
val scaledCcdata = scaler.fit(featuredCcdata).transform(featuredCcdata)

scaledCcdata.show(5)

// COMMAND ----------

// MAGIC %md
// MAGIC #### Process Categorical Data

// COMMAND ----------

val va_cat = new VectorAssembler()
    .setInputCols(colCat)
    .setOutputCol("features_cat")
val featuredCcdataCat = va_cat.transform(renamedCcdata)


featuredCcdataCat.show(5)

// COMMAND ----------

// create new column names
val newColCat = ArrayBuffer[String]()

for (c <- colCat) {
  val nc = c+"_idx"
  newColCat+=nc
}

val onehotColCat = ArrayBuffer[String]()

for (c <- colCat) {
  val nc = c+"_onehot"
  onehotColCat+=nc
}

// COMMAND ----------

// string indexer
val indexer = new StringIndexer()
    .setInputCols(colCat)
    .setOutputCols(newColCat.toArray)
val idxCcdata = indexer.fit(renamedCcdata).transform(renamedCcdata)

// COMMAND ----------

// one hot encoding
val encoder = new OneHotEncoder()
    .setInputCols(newColCat.toArray)
    .setOutputCols(onehotColCat.toArray)
val ohCcdata = encoder.fit(idxCcdata).transform(idxCcdata)

ohCcdata.show(5)

// COMMAND ----------

// MAGIC %md
// MAGIC #### Create Pipeline

// COMMAND ----------

val numPipeline = new Pipeline()
    .setStages(Array(va, scaler))
val catPipeline = new Pipeline()
    .setStages(Array(indexer, encoder))
val pipeline = new Pipeline().setStages(Array(numPipeline, catPipeline))
val newCcdata = pipeline.fit(renamedCcdata).transform(renamedCcdata)

newCcdata.show(5)

// COMMAND ----------

// MAGIC %md
// MAGIC #### Machine Learning Modeling

// COMMAND ----------

// vector assembler for features and label
onehotColCat+="scaled"
val lastCcdata = newCcdata.drop("features")

val va2 = new VectorAssembler()
    .setInputCols(onehotColCat.toArray)
    .setOutputCol("features")

val dataset = va2
    .transform(lastCcdata)
    .select("features", "label")

dataset.show(1, false)

// COMMAND ----------

// train test split
val Array(trainSet, testSet) = dataset.randomSplit(Array(0.8, 0.2))

trainSet.show(5)
testSet.show(5)

// COMMAND ----------

// function for evaluation metrics
def evaluation_metrics(predictions: DataFrame) : Unit = {
    val evaluator = new MulticlassClassificationEvaluator()
      .setPredictionCol("prediction")
      .setLabelCol("label")

    val evaluator2 = new BinaryClassificationEvaluator()
        .setRawPredictionCol("prediction")
        .setLabelCol("label")
        .setMetricName("areaUnderROC")

    val accuracy = evaluator.setMetricName("accuracy").evaluate(predictions)
    val f1_score = evaluator.setMetricName("f1").evaluate(predictions)
    val precision = evaluator.setMetricName("weightedPrecision").evaluate(predictions)
    val recall = evaluator.setMetricName("weightedRecall").evaluate(predictions)
    val auc = evaluator2.evaluate(predictions)


    println("Metrics on test data")
    println("-"*30)
    println(s"Accuracy: $accuracy")
    println(s"F1 Score: $f1_score")
    println(s"Precision: $precision")
    println(s"Recall: $recall")
    println(s"Area Under ROC Curve: $auc")
}

// COMMAND ----------

// MAGIC %md
// MAGIC #### Logistic Regression

// COMMAND ----------

// logistic regression
import org.apache.spark.ml.classification.{DecisionTreeClassifier, RandomForestClassifier, LogisticRegression}
import org.apache.spark.ml.tuning.{ParamGridBuilder, CrossValidator}
val lr = new LogisticRegression()
    .setMaxIter(50)
    .setFeaturesCol("features")
    .setLabelCol("label")

val paramGrid = new ParamGridBuilder()
    .addGrid(lr.regParam, Array(0.01, 0.05))
    .addGrid(lr.elasticNetParam, Array(0.01, 0.05))
    .build()

val evaluator = new BinaryClassificationEvaluator()

val cv = new CrossValidator()
    .setEstimator(lr)
    .setEstimatorParamMaps(paramGrid)
    .setEvaluator(evaluator)
    .setNumFolds(3)

val lrModel = cv.fit(trainSet)

print(lrModel.bestModel.extractParamMap())
// make predictions on the test data
val predictions = lrModel.transform(testSet)
evaluation_metrics(predictions)

// COMMAND ----------

// MAGIC %md
// MAGIC #### Decision Tree

// COMMAND ----------

// Instantiate the decision tree classifier
val dt = new DecisionTreeClassifier()
    .setLabelCol("label")
    .setFeaturesCol("features")

val paramGrid = new ParamGridBuilder()
    .addGrid(dt.maxDepth, Array(5, 6, 7))
    .build()

val evaluator = new BinaryClassificationEvaluator()

val cv = new CrossValidator()
    .setEstimator(dt)
    .setEstimatorParamMaps(paramGrid)
    .setEvaluator(evaluator)
    .setNumFolds(3)

val dtModel = cv.fit(trainSet)

// make predictions on the test data
val predictions = dtModel.transform(testSet)

evaluation_metrics(predictions)

// COMMAND ----------

// MAGIC %md
// MAGIC #### Random Forest

// COMMAND ----------

val rf = new RandomForestClassifier()
    .setNumTrees(50)
    .setLabelCol("label")
    .setFeaturesCol("features")

val paramGrid = new ParamGridBuilder()
    .addGrid(rf.featureSubsetStrategy, Array("auto", "all", "sqrt"))
    .addGrid(rf.minInstancesPerNode, Array(1, 2, 3))
    .build()

val evaluator = new BinaryClassificationEvaluator()
val cv = new CrossValidator()
    .setEstimator(rf)
    .setEstimatorParamMaps(paramGrid)
    .setEvaluator(evaluator)
    .setNumFolds(3)


val rfModel = cv.fit(trainSet)

// make predictions on the test data
val predictions = rfModel.transform(testSet)

evaluation_metrics(predictions)

// COMMAND ----------

// MAGIC %md
// MAGIC 
// MAGIC We have tried three different ML models: Logistic Regression, Decision Tree and Random Forest classifier, and evaluated with accuracy, F1 score, precision, recall and area under ROC curve (AUC). It is important to evaluate a classification problem with different metrics, since the accuracy is not sufficient to evaluate a model. If the data is unbalanced, a good accuracy score does not mean that the performance of the model is good. We need to check other metrics, especially AUC.
// MAGIC 
// MAGIC AUC tells us that how good is the separation between the classes. When AUC is close to 1, it means that all the classes are predicted correctly, while 0 AUC value is representing that all the classes are predicted wrong. AUC ~0.5 tells that all the classes are predicted randomly.
// MAGIC 
// MAGIC F1 score can be defined as 
// MAGIC ![alt text](https://latex.codecogs.com/gif.latex?F1=%5Cfrac%7B2%5Ctimes%7Bprecision%7D%5Ctimes%7Brecall%7D%7D%7Bprecision%20&plus;%20recall%7D)
// MAGIC 
// MAGIC Precision = (TP)/(TP+FP)
// MAGIC Recall = (TP)/(TP+FN)
// MAGIC 
// MAGIC where TP, TN, FP, and FN are true positives, true negatives, false positives, and false negatives respectively.
// MAGIC 
// MAGIC So that we can say that F1 score reaches its best value at 1 and worst score at 0.
// MAGIC 
// MAGIC All three of the models have similar accuracies:
// MAGIC 
// MAGIC #### LR:
// MAGIC Accuracy: 0.8230
// MAGIC 
// MAGIC F1 Score: 0.7972
// MAGIC 
// MAGIC AUC: 0.6437
// MAGIC 
// MAGIC #### DT:
// MAGIC Accuracy: 0.8207
// MAGIC 
// MAGIC F1 Score: 0.7929
// MAGIC 
// MAGIC AUC: 0.6362
// MAGIC 
// MAGIC #### RF:
// MAGIC Accuracy: 0.8135
// MAGIC 
// MAGIC F1 Score: 0.7772
// MAGIC 
// MAGIC AUC: 0.6097
// MAGIC 
// MAGIC By looking at the scores, we can say that logistic regression and decision tree models are giving more or less the same values, while random forest is slightly lower than them. Logistic regression is the winner with a tiny difference.
// MAGIC 
// MAGIC In order to increase the performance, feature selection algorithms can be used and PCA can be applied to numeric features.

// COMMAND ----------


