// Databricks notebook source
import org.apache.spark.ml.regression.LinearRegression

// COMMAND ----------

import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)

// COMMAND ----------

val df = spark.read.option("inferSchema", "true").option("header", "true").option("multiline",true).csv("/FileStore/tables/USA_Housing.csv")

// COMMAND ----------

df.show(1)

// COMMAND ----------

df.printSchema

// COMMAND ----------

df.select("Avg Area Income", "Avg Area House Age").show

// COMMAND ----------

// scala ml takes two columns 1 is the label/labels column and the other is a features column with all  features combines into 1 vector

import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors

// COMMAND ----------

val feature_cols = df.columns.slice(0,5).filter(!_.contains("Avg Area Number of Bedrooms"))


// COMMAND ----------

// excluding Average # of BedRooms as Python analysis shows potential multi-collinearity

val df_ml = df.select(df("Price").as("label"), $"Avg Area Income", $"Avg Area House Age", $"Avg Area Number of Rooms",  $"Area Population")

// COMMAND ----------

// Vector Assembler

val output = new VectorAssembler().setInputCols(feature_cols).setOutputCol("features").transform(df_ml).select($"label", $"features")

// COMMAND ----------

//val output = assember.transform(df_ml).select($"label", $"features")

// COMMAND ----------

output.show

// COMMAND ----------

val lr = new LinearRegression()
val lrModel = lr.fit(output)
val trainingSummary = lrModel.summary


// COMMAND ----------

println(f"R2       : ${trainingSummary.r2}")
println(f"R2_Adj   : ${trainingSummary.r2adj}")
println(f"Features : ${feature_cols.toList}")
println(f"Pvalues  : ${trainingSummary.pValues.toList}")
println(f"MAE      : ${trainingSummary.meanAbsoluteError}")
println(f"RMSE     : ${trainingSummary.rootMeanSquaredError}")
// results very similar to Python//
trainingSummary.predictions.show

// COMMAND ----------

// COMPARISON WITH GLM MODEL BELOW //

// COMMAND ----------

import org.apache.spark.ml.regression.GeneralizedLinearRegression

var glr = new GeneralizedLinearRegression()
  .setFamily("gaussian")
  .setLink("identity")
  .setMaxIter(10)
  .setRegParam(0.3)

// Fit the model
var model = glr.fit(output)

// Print the coefficients and intercept for generalized linear regression model
println(s"Coefficients: ${model.coefficients}")
println(s"Intercept: ${model.intercept}")

// Summarize the model over the training set and print out some metrics
var summary = model.summary

println(s"Coefficient Standard Errors: ${summary.coefficientStandardErrors.mkString(",")}")
// println(s"T Values: ${summary.tValues.mkString(",")}")
// println(s"P Values: ${summary.pValues.mkString(",")}")
// println(s"Dispersion: ${summary.dispersion}")
// println(s"Null Deviance: ${summary.nullDeviance}")
// println(s"Residual Degree Of Freedom Null: ${summary.residualDegreeOfFreedomNull}")
// println(s"Deviance: ${summary.deviance}")
// println(s"Residual Degree Of Freedom: ${summary.residualDegreeOfFreedom}")
println(s"AIC: ${summary.aic}")
// println("Deviance Residuals: ")
// summary.residuals().show()
var glm_preds = summary.predictions

glm_preds.show




// COMMAND ----------

import spark.implicits._
import org.apache.spark.sql.functions._
//calculated RMSE AND MAE from Predictions
var glm_rmse  = glm_preds.select(pow((sum(pow($"prediction" - $"label", 2)))/glm_preds.count, 0.5))
var glm_mae   = glm_preds.select(sum(abs($"prediction" - $"label"))/glm_preds.count)

println(f"GLM RMSE ${glm_rmse.head()(0)}")
println(f"GLM MAE ${glm_mae.head()(0)}")
