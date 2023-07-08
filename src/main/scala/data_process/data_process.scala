package data_process

import java.io.{BufferedWriter, File, FileWriter}

class data_process {

  val basePath = new File("src/main/resources/data/")
  val trainingFiles = new File(basePath, "train/")
  val testFiles = new File(basePath, "test/")

  val inputString = new File(basePath, "yahoo_stock.csv")
  def generate_data(mode: Boolean): Unit = {
    val date = if(mode) 5 else 6

    val Features = getFrature(inputString.getAbsolutePath).toList
    val label: List[String] = getLabel(inputString.getAbsolutePath).toList

    val Features_week = Features.drop(1).sliding(date).toList
    val label_week = label.drop(date - 1).dropRight(1)

    val data = Features_week zip label_week

    val numExamples = data.length
    val splitPos = math.ceil(numExamples * 0.7).toInt
    val (train, test) = data splitAt splitPos

    if(mode){
      writeCSV(train, trainingFiles.getAbsolutePath, date)
      writeCSV(test, testFiles.getAbsolutePath, date)
    }else{
      writeClassifyCSV(train, trainingFiles.getAbsolutePath)
      writeClassifyCSV(test, testFiles.getAbsolutePath)
    }

  }

  def writeCSV(batches: List[(List[Array[String]], String)], pathname: String, date: Int) = {
    for ((batch, index) <- batches.zipWithIndex) {
      val fileName = new File(pathname + s"/${index}.csv")
      val bw = new BufferedWriter(new FileWriter(fileName))
      bw.write(batch._2 + ",")

      for (i <- 0 to 4) {
        val data = batch._1(i)
        for (j <- 1 to 5) {
          bw.write(data(j))
          if (j != 5 || i != 4)
            bw.write(",")
        }
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

      val pre = batch._1(0)
      for(i <- 1 to 5) {
        val data = batch._1(i)
        for (j <- 1 to 5) {
          val dif = (data(j).toDouble - pre(j).toDouble).toString
          bw.write(dif)
          if(j != 5 || i != 5)
            bw.write(",")
        }
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
    } yield cols(3)
  }
}
