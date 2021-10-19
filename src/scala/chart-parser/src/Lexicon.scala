import scala.collection.mutable.{Set, HashMap}
import scala.io.Source

class Lexicon {

  val vocab: HashMap[String, Set[String]] = new HashMap

  val cat0 = ("Adj", Source.fromFile("categories/Adjective.txt").getLines.toArray)
  val cat1 = ("Adv", Source.fromFile("categories/Adverb.txt").getLines.toArray)
  val cat2 = ("Con", Source.fromFile("categories/Conjunction.txt").getLines.toArray)
  val cat3 = ("Det", Source.fromFile("categories/Determiner.txt").getLines.toArray)
  val cat4 = ("Inj", Source.fromFile("categories/Interjection.txt").getLines.toArray)
  val cat5 = ("N", Source.fromFile("categories/Noun.txt").getLines.toArray)
  val cat6 = ("P", Source.fromFile("categories/Preposition.txt").getLines.toArray)
  val cat7 = ("Pro", Source.fromFile("categories/Pronoun.txt").getLines.toArray)
  val cat8 = ("V", Source.fromFile("categories/Verb.txt").getLines.toArray)
  val categories = List(cat0, cat1, cat2, cat3, cat4, cat5, cat6, cat7, cat8)

  for (cat <- categories) {
    for (word <- cat._2) {
      if (vocab.contains(word)) {
        val tmp = vocab.get(word).get
        vocab.put(word, tmp union Set(cat._1))
      } else vocab += (word -> Set(cat._1))
    }
  }
}