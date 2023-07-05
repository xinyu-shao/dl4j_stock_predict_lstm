package data_process

import java.io.{BufferedWriter, File, FileWriter}

object data_process {
  def main(args: Array[String]): Unit = {
    generate_data()
  }
  def generate_data(): Unit ={
   val basePath = new File("src/main/resources/data/")
   val trainingFiles = new File(basePath, "train/")
   val testFiles = new File(basePath, "test/")

   val inputString = new File(basePath, "yahoo_stock.csv")
   val Features: List[String] = getFrature(inputString.getAbsolutePath).toList
   val label: List[String] = getLabel(inputString.getAbsolutePath).toList

    val Features_week = Features.sliding(7).toList
    val label_week = label.drop(8)
    val data = Features_week zip label_week

   val numExamples = data.length
   val splitPos = math.ceil(numExamples * 0.7).toInt
   val (train, test) = data splitAt splitPos

    writeCSV(train, trainingFiles.getAbsolutePath)
    writeCSV(test, testFiles.getAbsolutePath)
  }

  def writeCSV(batches: List[(List[String],String)], pathname: String) = {
    for ((batch, index) <- batches.zipWithIndex) {
      val fileName = new File(pathname + s"/${index}.csv")
      val bw = new BufferedWriter(new FileWriter(fileName))
      bw.write(batch._2 + ",")
      bw.write(batch._1.mkString(","))
      bw.newLine()
      bw.close
    }
  }

  def getFrature(fileName: String) = {
    val bufferedSource = scala.io.Source.fromFile(fileName)
    for {
      line <- bufferedSource.getLines.drop(1)
      cols = line.split(",").map(_.trim).dropRight(1)
    } yield cols(3)
  }

  def getLabel(fileName: String) = {
    val bufferedSource = scala.io.Source.fromFile(fileName)
    for {
      line <- bufferedSource.getLines.drop(1)
      cols = line.split(",").map(_.trim)
    } yield cols(5)
  }
  }
