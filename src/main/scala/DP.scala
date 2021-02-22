/*
    Step #1 :  Important Required Libraries
*/
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.sql.SparkSession

import java.io.File
import scala.reflect.io.Directory

object DP {

  def main(args: Array[String]): Unit = {


    val vAR_spark = SparkSession.builder
      .master("local[*]")
      .appName("Distributed Training_Data Parallelism")
      .getOrCreate()
    /*
      Step #2 :  Import Training Data
   */
    val vAR_sales_df = vAR_spark.read
      .option("header", true)
      .option("inferSchema", "true")
      //.csv("hdfs://localhost:9820/test/DS.AI_Sales_Trans_Australia_csv.csv", "hdfs://localhost:9820/test/DS.AI_Sales_Trans_Canada_csv.csv") //5000+
      .csv("hdfs://localhost:9820/test/DS.AI_Sales_Trans_V3_big_csv.csv")

    /*
    Step #3 :  Set Partition count as needed
    */
    val vAR_no_of_partitions =  5

    val vAR_partitioned_filePath = "C:/Users/vengi/ML/partitioned/"

    val vAR_sales_df_part = vAR_sales_df.repartition(vAR_no_of_partitions)

    val vAR_directory = new Directory(new File(vAR_partitioned_filePath))
    vAR_directory.deleteRecursively()
    vAR_sales_df_part.write.csv(vAR_partitioned_filePath)

    /*
   Step #4 :  Pre-process Training Data
    */
    val vAR_sales_df_speccols = vAR_sales_df_part
      .select("CustomerID", "ProductID", "TerritoryID", "Quantity", "OrderQty", "LineTotal")

    /*
    Step #5 : VectorAssembler is a transformer as it takes the input dataframe and returns the transformed dataframe
              with a new column which is vector representation of all the features.
    */
    val vAR_vectorassembler = new VectorAssembler()
      .setInputCols(Array("CustomerID", "ProductID", "TerritoryID", "Quantity", "OrderQty", "LineTotal"))
      .setOutputCol("features")

    val vAR_output = vAR_vectorassembler.transform(vAR_sales_df_speccols)
    vAR_output.select("features", "LineTotal").show(false)
    val vAR_labelcols = vAR_output.select("features", "LineTotal")

    /*
    Step #6 :Rename linetotal as label, now training data has two cols Features, Label only
    */

    val vAR_datatransformed = vAR_labelcols.withColumnRenamed("LineTotal", "label")
    val vAR_splitupdata = vAR_datatransformed.randomSplit(Array(0.80, 0.20))
    val vAR_training_data = vAR_splitupdata(0)
    val vAR_test_data = vAR_splitupdata(1)

    val vAR_lr = new LinearRegression()
    .setMaxIter(10)
    .setRegParam(0.3)
    .setElasticNetParam(0.8)
    .setFeaturesCol("features")
    .setLabelCol("label")

    /*
    Step #7 : Train Data
    */
    val vAR_lrmodel = vAR_lr.fit(vAR_training_data)

    /*
    Step #8 :  Predict on Test Data
    */
    val vAR_predictions = vAR_lrmodel.transform(vAR_test_data)
    vAR_predictions.show()

    println(s"Coefficients: ${vAR_lrmodel.coefficients} Intercept: ${vAR_lrmodel.intercept}")

    /*
    Step #9 :Summarize the model over the training set and print out some metrics
    */
    val vAR_trainingSummary = vAR_lrmodel.summary
    println(s"numIterations: ${vAR_trainingSummary.totalIterations}")
    println(s"objectiveHistory: [${vAR_trainingSummary.objectiveHistory.mkString(",")}]")
    vAR_trainingSummary.residuals.show()
    println(s"RMSE: ${vAR_trainingSummary.rootMeanSquaredError}")
    println(s"r2: ${vAR_trainingSummary.r2}")
    /*
    Step #10 : Uncomment System.in.read command and comment vAR_spark.stop() to verify at Spark Jobs
      System.in.read
    */
    vAR_spark.stop()


}
}