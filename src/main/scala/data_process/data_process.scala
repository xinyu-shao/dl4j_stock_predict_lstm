package data_process

import java.io.{BufferedWriter, File, FileWriter}

class data_process {

  val date = 7
  val basePath = new File("src/main/resources/data/")
  val trainingFiles = new File(basePath, "train/")
  val testFiles = new File(basePath, "test/")

  val inputString = new File(basePath, "yahoo_stock.csv")
  def generate_data(mode: Boolean): Unit = {

    val Features = getFrature(inputString.getAbsolutePath).toList
    val label: List[String] = getLabel(inputString.getAbsolutePath).toList

    val Features_week = Features.sliding(date).toList
    val label_week = label.drop(date)
    val data = Features_week zip label_week

    val numExamples = data.length
    val splitPos = math.ceil(numExamples * 0.7).toInt
    val (train, test) = data splitAt splitPos

    if(mode){
      writeCSV(train, trainingFiles.getAbsolutePath)
      writeCSV(test, testFiles.getAbsolutePath)
    }else{
      writeClassifyCSV(train, trainingFiles.getAbsolutePath)
      writeClassifyCSV(test, testFiles.getAbsolutePath)
    }

  }

  def writeCSV(batches: List[(List[Array[String]], String)], pathname: String) = {
    for ((batch, index) <- batches.zipWithIndex) {
      val fileName = new File(pathname + s"/${index}.csv")
      val bw = new BufferedWriter(new FileWriter(fileName))
      bw.write(batch._2 + ",")
//
//      bw.write(batch._1.mkString(","))
//      bw.newLine()
//      bw.close
      var count = date
      for(a <- batch._1){
        count -= 1
        bw.write(a.mkString(","))
        if(count > 0)
          bw.write(",")
        else
          count = date
      }
      bw.newLine()
      bw.close()
    }
  }

  def writeClassifyCSV(batches: List[(List[Array[String]], String)], pathname: String) = {

    for ((batch, index) <- batches.zipWithIndex) {
      val fileName = new File(pathname + s"/${index}.csv")
      val bw = new BufferedWriter(new FileWriter(fileName))
      val flag = if(batch._2.toDouble - batch._1.last(3).toDouble > 0 ) 1 else 0

      bw.write(flag + ",")

      var count = date
      val pre = batch._1(1).clone()
      for (a <- batch._1) {
        for(i <- 0 to 4){
          a(i) = (a(i).toDouble - pre(i).toDouble).toString
        }
        count -= 1
        bw.write(a.mkString(","))
        if (count > 0)
          bw.write(",")
        else
          count = date
      }
      bw.newLine()
      bw.close()
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
