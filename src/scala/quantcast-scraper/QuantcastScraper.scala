/**
  * Created by seanbethard on 7/17/16.
  */


import org.json._
import net.ruippeixotog.scalascraper.browser.JsoupBrowser
import net.ruippeixotog.scalascraper.dsl.DSL._
import net.ruippeixotog.scalascraper.dsl.DSL.Extract._
import net.ruippeixotog.scalascraper.model.Element

class QuantcastScraper {

  /** Returns demographic data for a site as JSON object
    *
    * • Gender: Male, Female
    * • Age: < 18, 18-24, 25-34, 35-44, 45-54, 55-64, 65+
    * • Male age: Male 18-24, Male 25-34, Male 35-44, Male 45-54, Male 55-64, Male 65+
    * • Female age: Female 18-24, Female 25-34, Female 35-44, Female 45-54, Female 55-64, Female 65+
    * • Income: $0-50k, $50-100k, $100-150k, $150k+
    * • Education: No College, College, Grad School
    * • Children: Kids, No Kids
    * • Ethnicity: Caucasian, Hispanic, African American, Asian
    * • Political Preference: Republican, Democrat, Independent
    *
    * */
  def getJsonObject(site: String): JSONObject = {

    val dataObject: JSONObject = new JSONObject

    try {

      val browser = JsoupBrowser()
      val doc = browser.get("https://www.quantcast.com/"+site+"#demographicsCard")
      val demoElement = doc >> extractor("script[key=demographicsData]", element)
      val politicsElement = doc >> extractor("script[key=politicalData]", element)

      dataObject.put("site", site)
      addPairs(dataObject, demoElement.innerHtml)
      addPairs(dataObject, politicsElement.innerHtml)


    } catch {
      case e: NoSuchElementException => println("DATA NOT THERE")
    }

    dataObject


  }

  /** Create custom JSON object from Quantcast data */
  def addPairs(myObject: JSONObject, rawJSON: String) {

    try{

      val webObj: JSONObject = new JSONObject(rawJSON)
      val webArr: JSONArray = webObj.getJSONArray("WEB")

      for (i <- 1 to webArr.length()) {
        val obj: JSONObject = webArr.getJSONObject(i-1)
        val bars: JSONArray = obj.getJSONArray("bars")

        for (j <- 1 to bars.length()) {
          {
            val title: String = bars.getJSONObject(j-1).getString("title")
            val index: Int = bars.getJSONObject(j-1).getInt("index")
            myObject.put(title, index)
          }
        }
      }


    } catch {
      case e: JSONException => println("DATA THERE BUT NOT PUBLIC")
    }



  }

  /** Returns list of quantified, unhidden Quantcast sites within page range */
  def getTopSites(start: Int, end: Int): List[String] = {

    var topSites = List[String]()

    for (i <- start to end) {
      val browser = JsoupBrowser()
      val doc = browser.get("https://www.quantcast.com/top-sites/"+i.toString)
      val leftHalf: List[Element] = doc >> extractor(".left-half tbody tr", elementList)
      val rightHalf: List[Element] = doc >> extractor(".right-half tbody tr", elementList)
      val rows: List[Element] = leftHalf ::: rightHalf

      for (j <- 1 to rows.size) {
        val badge: Element = rows(j-1) >> extractor("td.badge", element)
        val link: Element = rows(j-1) >> extractor("td.link", element)
        if (badge.text == "" && link.text != "Hidden profile") {
          topSites = link.text :: topSites
        }
      }
    }
    topSites
  }
}