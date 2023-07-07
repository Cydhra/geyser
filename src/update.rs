use std::str::FromStr;
use std::time::Duration;
use isahc::HttpClient;
use isahc::config::RedirectPolicy;
use isahc::cookies::CookieJar;
use isahc::http::Uri;
use isahc::prelude::*;
use scraper::{Html, Selector};
use crate::database::Database;

const WIKI_URI: &str = "https://scp-wiki.wikidot.com/";
const VOTE_ENDPOINT: &str = "https://scp-wiki.wikidot.com/ajax-module-connector.php";
const USER_AGENT: &str = "geyser-scp-vote-counter/0.2.0";

/// Runs the update process. It downloads articles and votes from the wiki and serializes them into
/// a database file which can be loaded by the main program.
pub(crate) struct Updater {
    /// The database builder.
    database: Database,
    client: HttpClient,
    cookie_jar: CookieJar,
    head_selector: Selector,
    script_selector: Selector,
}

impl Updater {
    pub fn new() -> Self {
        let cookie_jar = CookieJar::new();

        Self {
            database: Database::new(),
            client: HttpClient::builder()
                .timeout(Duration::from_secs(5))
                .default_header("User-Agent", USER_AGENT)
                .redirect_policy(RedirectPolicy::Follow)
                .cookie_jar(cookie_jar.clone())
                .build()
                .unwrap(),
            cookie_jar,
            head_selector: Selector::parse("head").unwrap(),
            script_selector: Selector::parse("script").unwrap(),
        }
    }

    /// Scrape SCP articles and user votes from the wiki without the API. Stores them in a
    /// database file named ```database.bin```.
    pub(crate) fn update(&mut self, from: usize, to: usize) {
        println!("Updating database...");

        let div_selector = Selector::parse("div").unwrap();
        let span_selector = Selector::parse("span").unwrap();
        let ref_selector = Selector::parse("a").unwrap();

        // I am unsure what this token is even used for, but it is required to access modules.
        // It is obtained by loading any wiki page and extracting it from the cookies.
        // It is an access token for the current session, and since this bot is not logged in, it
        // is a guest token with low permissions. Why this is necessary to access the vote module
        // is beyond me, since any session gets one automatically.
        println!("Obtaining wiki_token7...");
        self.client.head(WIKI_URI).unwrap();
        let wiki_token7 = self.cookie_jar.get_by_name(&Uri::from_str(WIKI_URI).unwrap(), "wikidot_token7").unwrap().value().to_owned();
        println!("wiki_token7: {}", wiki_token7);

        // for now this cannot handle non-scp-articles. It is easy to add by changing the for loop
        // to use a pre-computed list of article names.
        for number in from..=to {
            // download article
            let article_name = format!("scp-{:03}", number);
            let article = self.download_article(&article_name);
            let body = if let Some(body) = article {
                body
            } else {
                continue;
            };

            // parse article dom and extract article id
            let dom = Html::parse_document(&body);
            let page_id = if let Some(page_id) = self.extract_page_id(&dom) {
                page_id.to_string()
            } else {
                println!("Failed to extract page id for article {}", article_name);
                continue;
            };

            // download votes
            print!("page id: {}. ", page_id);
            let votes = if let Some(votes) = self.get_votes(&page_id, &wiki_token7) {
                votes
            } else {
                println!("Failed to download votes for article {}", article_name);
                continue;
            };

            // parse vote answer
            let answer = if let Some(answer) = json::parse(&votes).ok() {
                answer
            } else {
                println!("Failed to parse vote answer for article {}", article_name);
                continue;
            };

            let body = if let Some(body) = answer["body"].as_str() {
                body
            } else {
                println!("Failed to extract vote body for article {}", article_name);
                continue;
            };
            let dom = Html::parse_document(&body);
            let mut all_votes = dom.select(&div_selector).next().unwrap().select(&span_selector);

            // extract votes from answer
            let mut votes = Vec::new();
            while let Some(user_span) = all_votes.next() {
                let vote_span = if let Some(vote_span) = all_votes.next() {
                    vote_span
                } else {
                    println!("Failed to extract some votes for article {}", article_name);
                    break;
                };

                if let Some(user_name_html) = user_span.select(&ref_selector).nth(1) {
                    let user_name = user_name_html.inner_html().as_str().trim().to_owned();
                    let vote = vote_span.inner_html().as_str().trim().to_owned();

                    let user_id = self.database.add_user(user_name);
                    votes.push((user_id, vote == "+"));
                } // else: account deleted
            }

            // add article to database
            println!("added article to database with {} votes", votes.len());
            self.database.add_article(article_name, page_id, votes);
        }

        println!("Finished generating database. Saving to file...");
        self.database.save();
    }

    /// Make a request to the given url path and return the response body as a string.
    /// Returns None if the request failed.
    fn download_article(&self, article: &str) -> Option<String> {
        print!("Downloading article {}... ", article);
        let url = WIKI_URI.to_owned() + article;
        self.client.get(url).map_or(None, |mut response| {
            if response.status().is_success() {
                print!("success: ");
                response.text().map_or(None, |text| Some(text))
            } else {
                println!("failed: Error {}", response.status());
                None
            }
        })
    }

    /// Make a post request to the voting module url and return the response body as a
    /// string. Returns None if the request failed. Requires the page_id to request votes for
    /// and the wiki_token7 cookie.
    fn get_votes(&self, page_id: &str, wiki_token7: &str) -> Option<String> {
        let request_body = form_urlencoded::Serializer::new(String::new())
            .append_pair("pageId", page_id)
            .append_pair("moduleName", "pagerate/WhoRatedPageModule")
            .append_pair("callbackIndex", "1")
            .append_pair("wikidot_token7", wiki_token7)
            .finish();

        self.client.post(VOTE_ENDPOINT, request_body).map_or(None, |mut response| {
            if response.status().is_success() {
                response.text().map_or(None, |text| Some(text))
            } else {
                None
            }
        })
    }

    /// Extract the internal page id from the article by scraping it out of a javascript tag.
    fn extract_page_id(&self, article: &Html) -> Option<u32> {
        let header = article.select(&self.head_selector).next().unwrap();

        for script_tag in header.select(&self.script_selector) {
            if script_tag.value().attr("src").is_none() {
                let script_source = script_tag.first_child().unwrap().value().as_text().unwrap();
                if script_source.contains("WIKIREQUEST.info.pageId") {
                    return Some(u32::from_str_radix(script_source.split("WIKIREQUEST.info.pageId = ").nth(1).unwrap().split(';').next().unwrap(), 10).unwrap());
                }
            }
        }

        None
    }
}