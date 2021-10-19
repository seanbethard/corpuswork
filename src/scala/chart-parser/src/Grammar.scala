import scala.io.Source
import scala.collection.immutable.HashMap

class Grammar(path: String) {

  val entries = Source.fromFile(path).getLines.toList
  var productions: HashMap[List[String], String] = new HashMap

  entries foreach { entry =>
    val both = entry.split("->")
    val lhs = both(0).trim()
    val rhs = both(1).trim()
    val nts = rhs.split(" ").toList
    productions += (nts -> lhs)
  }
}