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

   val inputString = new File(basePath, "1.csv")
   val Features: List[Array[String]] = getFrature(inputString.getAbsolutePath).toList
   val label: List[String] = getLabel(inputString.getAbsolutePath).toList

    val Features_week = Features.sliding(7).toList
    val label_week = label.drop(8)
   val numExamples = Features_week.length
   val splitPos = math.ceil(numExamples * 0.7).toInt
   val (train_Feature, test_Feature) = Features_week splitAt splitPos

   writeFeatureCSV(train_Feature, trainingFiles.getAbsolutePath + "/train_Feature/")
    writeFeatureCSV(test_Feature, testFiles.getAbsolutePath + "/test_Feature/")

    val (train_label, test_label) = label_week splitAt splitPos

    writeLabelCSV(train_label, trainingFiles.getAbsolutePath + "/train_label/")
    writeLabelCSV(test_label, testFiles.getAbsolutePath + "/test_label/")
  }

  def writeFeatureCSV(batches: List[List[Array[String]]], pathname: String) = {
    for ((batch, index) <- batches.zipWithIndex) {
      val fileName = new File(pathname + s"${index}.csv")
      val bw = new BufferedWriter(new FileWriter(fileName))
      for(a <- batch){
        bw.write(a(3))
        bw.newLine()
      }
      bw.close
    }
  }

  def writeLabelCSV(batches: List[String], pathname: String) = {
    for ((batch, index) <- batches.zipWithIndex) {
      val fileName = new File(pathname + s"${index}.csv")
      val bw = new BufferedWriter(new FileWriter(fileName))
      bw.write(batch)
      bw.newLine()
      bw.close
    }
  }
  def getFrature(fileName: String) = {
    val bufferedSource = scala.io.Source.fromFile(fileName)
    for {
      line <- bufferedSource.getLines.drop(1)
      cols = line.split(",").map(_.trim).dropRight(1)
    } yield cols
  }

  def getLabel(fileName: String) = {
    val bufferedSource = scala.io.Source.fromFile(fileName)
    for {
      line <- bufferedSource.getLines.drop(1)
      cols = line.split(",").map(_.trim)
    } yield cols(5)
  }
  }
