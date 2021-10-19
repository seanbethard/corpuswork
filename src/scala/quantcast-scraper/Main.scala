import java.io.{BufferedWriter, File, FileWriter}

import org.json.JSONObject

import scala.io.Source


/**
  * Created by seanbethard on 7/17/16.
  */


object Main extends QuantcastScraper {


  /** Write List[String] to file line by line */
  def writeListLineByLine(list: List[String], fileName: String): Unit = {

    val file = new File(fileName)
    val bw = new BufferedWriter(new FileWriter(file))

    for (i <- 1 to list.size) {
      bw.write(list(list.size - i)+"\n")
    }
    bw.close()
  }

  /** Read file into list */
  def readFromFileIntoList(fileName: String): List[String] = {

    val listOfLines = Source.fromFile(fileName).getLines.toList
    listOfLines

  }



  /** Write top sites to file
    * set page range and output location */
  def writeSitesToFile(): Unit = {

    val t0 = System.nanoTime()
    val topSites: List[String] = getTopSites(50, 53)
    writeListLineByLine(topSites, "data/quantcast/sites/sites5.txt")

    // 1176 sites written to file in 3.77 seconds
    // 7826 sites written to file in 20.77 seconds
    val t1 = System.nanoTime()
    val elapsed = BigDecimal((t1 - t0) / 1000000000.0).setScale(2, BigDecimal.RoundingMode.HALF_UP).toDouble
    println(topSites.size+" sites written to file in " + elapsed + " seconds")

  }

  /** Write demographics to file
    * set input and output locations */
  def writeDemoToFile(): Unit = {

    val t0 = System.nanoTime()
    val input: String = "data/quantcast/sites/sites0.txt"
    val sites: List[String] = readFromFileIntoList(input)
    var demoList: List[String] = List[String]()

    for (i <- 1 to sites.length) {
      println(sites(i-1))
      val demoJSON: JSONObject = getJsonObject(sites(i-1))
      println(demoJSON)
      demoList = demoJSON.toString() :: demoList
    }
    writeListLineByLine(demoList, "data/quantcast/demographics/demo0.txt")

    //160 JSON objects written to file in 82.32 seconds
    val t1 = System.nanoTime()
    val elapsed = BigDecimal((t1 - t0) / 1000000000.0).setScale(2, BigDecimal.RoundingMode.HALF_UP).toDouble
    println(demoList.size+" JSON objects written to file in " + elapsed + " seconds")

  }

  /** Entry point */
  def main(args: Array[String]): Unit = {
//    writeSitesToFile()
    writeDemoToFile()
  }
}
