// Databricks notebook source
val  x = 100

// COMMAND ----------

var myvar: Int = 10
myvar

// COMMAND ----------

val c = 12.0
val str = "hello"
val MYVAL = "123.9087U"
/* string use double quotes */
println(c)

// COMMAND ----------

println("Modulus")
4%1

// COMMAND ----------

println("exponent")
math.pow(4,2)

// COMMAND ----------

val name = "Edwin"
val last_name = "T"
println(f"$name $last_name")

println(s"$name $last_name 2")

// COMMAND ----------

val st = "This is a string"
st.charAt(3) /* at index 3*/

// COMMAND ----------

st slice(0,4) /*inclusive , exclusive*/

// COMMAND ----------

st contains("is a")

// COMMAND ----------

st contains("hello")

// COMMAND ----------

/* tuple */
val tup = (1,2,3, "hello")
/* index starts at 1 , use _to get values by index */
val nested = (1,3, 4 , (5,6,7))
nested._4._1

// COMMAND ----------

var e = "mutable"
e = "showing it"
print(e)


// COMMAND ----------

var f = "immutable"
f = try{"showing it"} catch{
 
    case ex:reassignment to val => ""
  }
  println(f)

// COMMAND ----------

/* very important thing  on Division*/ 
/* use floats unless you want a floor value*/
println(1.0/2.0 == 1/2)
println(1.0/2.0)
println(1/2)

// COMMAND ----------

println(math.pow(2,5))
println(180%7)
val pet_name = "Sammy"
println(f"My dog's name is $pet_name")
val x = "sadfjshfjyuyxyzjkfuidkjklhasyysdfk"
println(x contains("xyz"))
/* val is imutable */
val new_tup = (1,2,3,(4,5,6))
println(f"6 is at ${new_tup._4._3}")

// COMMAND ----------

val alist =List(1,2,3,4,5 ,"another type")
/* lists start at 0 index*/
alist(0)

// COMMAND ----------

println(alist.head)
println(alist.tail)
println(f"Nested List : ${List(List(1,2,3), 4,5,6)}")
/** sorted can only be for same type*/
//println(f"Sorted List : ${alist.sorted}") causes an error
val sameType = List(88, 99, 0, 78,90, 65)
println(f"Sorted - Same Type: ${sameType.sorted}")

// COMMAND ----------

// some list operations, lists are immutabale //
println(alist.size)
val num_list = alist.slice(0, alist.size -1)
println(num_list)
//println(num_list.sum)

// COMMAND ----------

/** these get the last value , similar to -1 index in python*/
println(alist.drop(5))
println(alist.takeRight(1))

// COMMAND ----------

//arrays are MUTABLE //
val arr = Array(1, 2, 3)
val arr2 = Array.range(0, 10)
// using Array.range with step size 
val arr3 = Array.range(0, 10, 2)
// IMPORTANT Range method created an immutable "Range" object //
val rangeObj = Range(0,10)

// COMMAND ----------

// sets //
// imutable , unordered collection//
val s = Set(1,22,5,5, 0,0, 7,8,9, 7)
println(s)

//mutable set//
val imuts = scala.collection.mutable.Set(1, 3, 8, 9, 1)
println(imuts)
imuts.add(100)
println(imuts)
print(imuts.min)
//list to set 
println(alist.toSet)

// COMMAND ----------

//Maps  , like python dicts 

val amap = Map(("a", 1), ("b", 2)) //immutable map 
val mutmap = scala.collection.mutable.Map(("a", 1), ("b", "b"))
println(mutmap)
println(mutmap get "a")
println(mutmap("a"))
//editing value
mutmap += ("a" -> 111)
// adding pair 
mutmap += ("c" -> "222")
println(mutmap.keys)
println(mutmap.values)


// COMMAND ----------

var list2 = List(1,2,3,4,5)
println(list2.contains(5))
println(list2.sum)

val oddArr = Array.range(1,16,2)
println(oddArr)

val list3 = List(2,3,1,4,5,6,6,1,2)
//unique elements 
println(list3.toSet)



val mutmap2 = scala.collection.mutable.Map(("Sammy", 3),
("Frankie", 7),
("John", 45))

println(mutmap2.keys)
println(mutmap2.values)

mutmap2 +=("Mike" -> 27)
println(mutmap2)

// COMMAND ----------

// if else, summilar to javascript 

if (1 ==1 || 2==2 ){ 
  println("Yes")
  } 
  else if ((3==3) && (4==4)){ 
    println("Yes")
    }
  else{

      println("idkn")
    }

// COMMAND ----------

for(i <- Array.range(0,9)){
   println(i)
}

// COMMAND ----------

// import the break from util.control.Breaks._
import util.control.Breaks._
var x = 0

while(x<5) {

println("okay")
x = x+1

}

while(x<10) {

println("okay")
x = x+1
if (x==6) break

}



// COMMAND ----------

def simple():Unit ={
  

  println("test")
}


simple()

// COMMAND ----------

def arithm(num:Double, num2:Double, func:String): Double = {
     val funcL = func.toLowerCase.trim
     if (funcL.toLowerCase == "divide"){
         return num/num2
     } 
     else if (funcL=="add"){
      return num + num2
     }
     else if (funcL =="subtract"){
      return math.abs(num-num2)
     } 
     else {
      return math.abs(num *num2)
     }
}

arithm(5,11, "subtract")

// COMMAND ----------

def variance( data: List[Double],ddof: Int ):(Any, Any) = {
    val n = data.size
    if (n == 1 && ddof>0){
      return (None, None)
    }
    val mean = data.sum/n
    val sosd = data.map(x => math.pow(x-mean, 2)).sum
    val varc = sosd/(n -ddof)
    val std  = math.pow(varc, 0.5)
    return (varc, std)
}

// COMMAND ----------

val unbiased_var = variance(List(1, 3), 1)


// COMMAND ----------

class SimpleVariance(var data: List[Double], var ddof: Int) {
  assert((data.size > 0 && ddof == 0 )|| (data.size > 1 && ddof==1), "List must not be empty")

  def variance(): Double = {
    val n = data.size
    val mean = data.sum/n
    val sosd = data.map(x => math.pow(x-mean, 2)).sum
    val varcalc = sosd/(n -ddof)
    val std  = math.pow(varcalc, 0.5)
    return varcalc
   } 
  var varc = variance()
  
}

/*class SimpleStd(var data: List[Double], var ddof: Int ) extends SimpleVariance(data: List[Double], ddof: Int ){
  override var varc = super.varc
  override def variance = super.variance
  var std = math.pow(varc, 0.5)
}*/

// COMMAND ----------

var stats = new SimpleVariance( List(1,2,3,4,5,6), 1)
stats.variance()
stats.varc

// COMMAND ----------



// COMMAND ----------

//dataframes
var df_fin = spark.read.option("inferSchema", "true").option("header", "true").csv("/FileStore/tables/CitiGroup2006_2008.csv")

// COMMAND ----------

df_fin.head(4)
for ( row <- df_fin.head(4)){

    println(row)

}

// COMMAND ----------

df_fin.columns

// COMMAND ----------

df_fin.describe().show()

// COMMAND ----------

df_fin.select($"Date", $"Open", $"High").show(5)

// COMMAND ----------

// add column
var df2 = df_fin.withColumn("HL",df_fin("High") - df_fin("Low") )
df2.printSchema()

// COMMAND ----------

// rename 
df2.select(df2("HL").as("HML"), $"High",$"Low" ).show(5)

// COMMAND ----------

//filtering 

val df3 = spark.read.option("inferSchema", "true").option("header", "true").csv("/FileStore/tables/CitiGroup2006_2008.csv")

// COMMAND ----------

df3.printSchema

// COMMAND ----------

df3.select($"Date", $"Close").show(3)

// COMMAND ----------

df3.filter(df3("Close") > 500.00).count

// COMMAND ----------

df3.filter($"Close" > 500.00).count

// COMMAND ----------

//sql notation 
df3.filter("Close > 500.00").count

// COMMAND ----------

df3.filter("Close > 500.00 and Open < 500.00").count

// COMMAND ----------

df3.filter($"Close" > 500.00 && $"Open" < 500.00).count

// COMMAND ----------

df3.filter($"Close" > 500.00 && $"Open" < 500.00).describe().show()


// COMMAND ----------

df3.describe().show()

// COMMAND ----------

// save summary as object 
val summary = df3.describe().collect()
summary
//summary.filter($"summary" == "mean")

// COMMAND ----------

val df4 = spark.read.option("header", "true").option("inferSchema", "true").csv("/FileStore/tables/CitiGroup2006_2008.csv")

// COMMAND ----------

df4.filter($"High" === 400).show

// COMMAND ----------

import spark.implicits._
import org.apache.spark.sql.functions._
df4.select(corr("High", "Open")).show()

// COMMAND ----------

df4.select(max("High")).show()

// COMMAND ----------

val df5 = spark.read.option("inferSchema", "true").option("header", "true").csv("/FileStore/tables/Sales.csv")

// COMMAND ----------

df5.printSchema

// COMMAND ----------

// group by functions
var grp_cols           = Array("Company", "Person")
var rename_cols        = Array("Company", "Person", "Total_Sales", "Avg_Sales")
var agg_cols           = Array(sum("Sales"), avg("Sales"))
var order_by_cols      = Array($"Company", $"Person".desc, $"Total_Sales".desc)

df5.groupBy(grp_cols.head, grp_cols.tail:_*).agg(agg_cols.head, agg_cols.tail:_*).toDF(rename_cols:_*).orderBy(order_by_cols:_*).toDF().show


// COMMAND ----------

// Dynamic Group By Function

def dynamic_grpBy( df_in         : org.apache.spark.sql.DataFrame, 
                   grp_cols      : Array[String], 
                   agg_cols      : Array[org.apache.spark.sql.Column],
                   order_by_cols : Array[org.apache.spark.sql.Column]                
                 ): org.apache.spark.sql.DataFrame ={
  
      val df_out = df_in.groupBy(grp_cols.head, grp_cols.tail:_*).agg(agg_cols.head, agg_cols.tail:_*).toDF(rename_cols:_*).orderBy(order_by_cols:_*).toDF

      return df_out

}

// COMMAND ----------

var df6 = dynamic_grpBy(df5, grp_cols, agg_cols, order_by_cols)
df6.show

// COMMAND ----------

val df_nulls = spark.read.option("inferSchema", "true").option("header", "true").csv("/FileStore/tables/ContainsNull.csv")

df_nulls.printSchema

// COMMAND ----------

df_nulls.show

// COMMAND ----------

/// for na.drop(x) DROPPING ALL ROWS WITH A LESS THAN A MINIMUM OF x non-null values
df_nulls.na.drop(2).show

// COMMAND ----------

/// just  for practice 
var mean_stdev_sales = df_nulls.filter($"Sales" > 0).select(stddev_samp("Sales") + mean("Sales")).head()(0).toString.toDouble

// COMMAND ----------

//Dynamic missing values simple methods

var df_filled = df_nulls.na.fill("<UNK>", Array("Name")).na.fill(mean_stdev_sales, Array("Sales")).show

// COMMAND ----------

df2.withColumn("Today", current_date().as("Current_Date")).withColumn("curr_ts", current_timestamp()).as("current_ts").show

// COMMAND ----------

//get time zone
java.util.TimeZone.getAvailableIDs

// COMMAND ----------



// COMMAND ----------

//

spark.conf.get("spark.sql.session.timeZone")

// COMMAND ----------

df2.printSchema

// COMMAND ----------

var df_time = df2.withColumn("year", year($"Date"))

df_time.show(5)

// COMMAND ----------

// window functions 
import org.apache.spark.sql.expressions.Window
val windowSpec  = Window.partitionBy("year").orderBy("Date")
df_time.withColumn("row_number",row_number.over(windowSpec)).show(3)

// COMMAND ----------

val windowSpec  = Window.partitionBy("year").orderBy("Date")
df_time.withColumn("prev_close",lag("Close", 1).over(windowSpec)).show(3)

// COMMAND ----------

// x day moving window -- example is 7 day ma
val moving_window_spec = Window.orderBy("date").rowsBetween(-7, 0)
val moving_window_spec_30 = Window.orderBy("date").rowsBetween(-30, 0)
val df_citi = df_time.withColumn("MA7",avg("Close").over(moving_window_spec)).withColumn("MA30", avg("Close").over(moving_window_spec_30))

// COMMAND ----------

display(df_citi.select("Date", "Close", "MA7", ))

// COMMAND ----------

// DATAFRAME PROJECT
// Use the Netflix_2011_2016.csv file to Answer and complete
// the commented tasks below!


// Load the Netflix Stock CSV File, have Spark infer the data types.

var df_nflx = spark.read.option("inferSchema", "true").option("header", "true").csv("/FileStore/tables/Netflix_2011_2016.csv")

// What are the column names?
 println(f"columns ${df_nflx.columns.toList}")

// What does the Schema look like?

df_nflx.printSchema

// Print out the first 5 columns.

df_nflx.columns.toList.slice(0,5)

// Use describe() to learn about the DataFrame.
df_nflx.describe()

// Create a new dataframe with a column called HV Ratio that
// is the ratio of the High Price versus volume of stock traded
// for a day.
var df_prepped =  df_nflx.withColumn("hv_ratio", $"High"/$"Volume")

// What day had the Peak High in Price?
df_prepped.filter($"High" === df_prepped.select(max("High").as("max_high")).head()(0).toString.toDouble).show

// What is the mean of the Close column?
df_prepped.select(mean("Close").as("avg_close")).show

// What is the max and min of the Volume column?
df_prepped.select(min("Volume").as("min_volume"), max("Volume").as("max_volumne")).show

// For Scala/Spark $ Syntax

// How many days was the Close lower than $ 600?
df_prepped.filter($"Close" < 600.00).count

// What percentage of the time was the High greater than $500 ?
(df_prepped.filter($"High">500.00).count.toDouble/df_prepped.count.toDouble) * 100.00

// What is the Pearson correlation between High and Volume?
df_prepped.select(corr("High", "Volume")).head()(0)

// What is the max High per year?
df_prepped.withColumn("Year", year($"Date")).groupBy("Year").agg( max("High")).orderBy("Year").show

// What is the average Close for each Calender Month?
df_prepped.withColumn("Month", trunc($"Date", "Month")).groupBy("Month").agg(avg("Close").as("Monthly_Avg_Close")).orderBy("Month").show

