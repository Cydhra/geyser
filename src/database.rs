use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{Read, Write};
use std::ops::AddAssign;

/// Database of articles and user votes. This struct can be serialized to store it.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct Database {
    /// Maps article names to internal article ids.
    articles: BTreeMap<String, usize>,

    /// A list of all article page ids. Index in this list is the article id.
    page_ids: Vec<String>,

    /// A list of all article votes. Each entry is a tuple of the user id and the vote. Index in
    /// this list is the article id.
    article_votes: Vec<Vec<(usize, bool)>>,

    /// The total number of votes.
    total_votes: usize,

    /// A list of all user names. Second component is the user id.
    users: BTreeMap<String, usize>,
}

impl Database {
    /// Creates a new empty database builder.
    pub(crate) fn new() -> Self {
        Self {
            articles: BTreeMap::new(),
            page_ids: Vec::new(),
            article_votes: Vec::new(),
            total_votes: 0,
            users: BTreeMap::new(),
        }
    }

    /// Loads the database from file.
    pub(crate) fn load() -> Self {
        let mut file = File::open("database.bin").expect("Failed to open database file.");
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)
            .expect("Failed to read database from file.");
        serde_cbor::from_slice(&buffer).unwrap()
    }

    /// Saves the database to file.
    pub(crate) fn save(&self) {
        let mut file = File::create("database.bin").unwrap();
        let serialized = serde_cbor::to_vec(self).unwrap();
        file.write_all(&serialized)
            .expect("Failed to write database to file.");
    }

    /// Adds a new article and all its ratings to the database.
    /// # Parameters
    /// - ```article```: The article name.
    /// - ```page_id```: The wikidot page id of the article for future requests
    /// - ```votes```: A list of tuples of user ids and votes. The first component of the tuple is
    /// the user id, the second component is the vote (true for upvote, false for downvote).
    pub(crate) fn add_article(
        &mut self,
        article: String,
        page_id: String,
        votes: Vec<(usize, bool)>,
    ) {
        self.articles.insert(article, self.articles.len());
        self.page_ids.push(page_id);
        self.total_votes += votes.len();
        self.article_votes.push(votes);
    }

    pub(crate) fn update_article(&mut self, article: String, votes: Vec<(usize, bool)>) {
        let article_id = *self.articles.get(&article).unwrap();
        self.total_votes -= self.article_votes[article_id].len();
        self.total_votes += votes.len();
        self.article_votes[article_id] = votes;
    }

    /// Adds a new user to the database.
    /// Returns the user id.
    pub(crate) fn add_user(&mut self, user: String) -> usize {
        if let Some(id) = self.users.get(&user) {
            return *id;
        }
        let user_id = self.users.len();
        self.users.insert(user, user_id);
        user_id
    }

    /// Use linear regression to estimate a singular value decomposition of the user-vote matrix.
    /// The result is a prediction model that can be used to predict the votes of users for articles
    /// they have not yet voted on.
    pub(crate) fn train_prediction_model(
        self,
        latent_factors: usize,
        iterations: usize,
        learning_rate: f64,
        regularization: f64,
    ) {
        let mut user_factors =
            nalgebra::DMatrix::from_fn(self.users.len(), latent_factors, |_, _| 0.1);
        let mut article_factors =
            nalgebra::DMatrix::from_fn(self.articles.len(), latent_factors, |_, _| 0.1);

        for factor in 0..latent_factors {
            println!("Factor {}/{}", factor + 1, latent_factors);

            let now = std::time::Instant::now();
            let mut mean_square_error = 0.0;
            for _ in 0..iterations {
                let gradients = self
                    .article_votes
                    .par_iter()
                    .enumerate()
                    .fold(
                        || {
                            (
                                nalgebra::DVector::<f64>::zeros(self.users.len()),
                                nalgebra::DVector::<f64>::zeros(self.articles.len()),
                                0.0,
                                0,
                            )
                        },
                        |(
                            mut user_gradient,
                            mut article_gradient,
                            mut mean_square_error,
                            mut count,
                        ),
                         (article_id, votes)| {
                            for &(user_id, vote) in votes {
                                let vote = if vote { 1.0 } else { -1.0 };

                                let user_factor = user_factors.row(user_id);
                                let article_factor = article_factors.row(article_id);
                                let prediction = user_factor.dot(&article_factor);
                                let error = vote - prediction;
                                mean_square_error += error * error;

                                let user_factor_value = user_factor[factor];
                                let article_factor_value = article_factor[factor];
                                user_gradient[user_id].add_assign(
                                    article_factor_value * error
                                        - user_factor_value * regularization,
                                );
                                article_gradient[article_id].add_assign(
                                    user_factor_value * error
                                        - article_factor_value * regularization,
                                );
                            }
                            count += votes.len();
                            (user_gradient, article_gradient, mean_square_error, count)
                        },
                    )
                    .reduce(
                        || {
                            (
                                nalgebra::DVector::<f64>::zeros(self.users.len()),
                                nalgebra::DVector::<f64>::zeros(self.articles.len()),
                                0.0,
                                0,
                            )
                        },
                        |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2, a.3 + b.3),
                    );

                user_factors
                    .column_mut(factor)
                    .add_assign(learning_rate * gradients.0);
                article_factors
                    .column_mut(factor)
                    .add_assign(learning_rate * gradients.1);
                mean_square_error = gradients.2 / gradients.3 as f64;
            }

            println!("Factor finished in {}ms.", now.elapsed().as_millis());
            println!("Mean square error: {}", mean_square_error);
        }

        println!("Training finished.");

        println!("Constructing read-filter...");

        // Construct a user-to-article filter to remove predictions about articles the user
        // has already voted on.
        let mut user_votes = vec![Vec::new(); self.users.len()];
        for user in self.users.keys() {
            let user_id = self.users[user];
            let user_votes = &mut user_votes[user_id];

            for (article_id, votes) in self.article_votes.iter().enumerate() {
                if votes.iter().any(|(id, _)| *id == user_id) {
                    user_votes.push(article_id);
                }
            }
        }

        let model = PredictionModel {
            database: self,
            user_factors,
            article_factors,
            user_votes,
        };

        let mut file = File::create("prediction_model.bin").unwrap();
        let serialized = serde_cbor::to_vec(&model).unwrap();
        file.write_all(&serialized)
            .expect("Failed to write prediction model to file.");
        println!("Saved prediction model to file.");
    }

    /// Returns the internal wikidot page id for a given article or none, if the article is not
    /// in the database.
    pub fn get_page_id(&self, article: &str) -> Option<&String> {
        self.articles.get(article).map(|id| &self.page_ids[*id])
    }
}

/// A prediction model for the user votes. This is created from a database by training a linear
/// regression model to create the user_factors and article_factors matrices.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct PredictionModel {
    database: Database,
    user_factors: nalgebra::DMatrix<f64>,
    article_factors: nalgebra::DMatrix<f64>,
    user_votes: Vec<Vec<usize>>,
}

impl PredictionModel {
    /// Loads the prediction model from file ``prediction_model.bin``.
    pub fn load() -> Self {
        let mut file =
            File::open("prediction_model.bin").expect("Failed to open prediction model file.");
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)
            .expect("Failed to read prediction model from file.");
        serde_cbor::from_slice(&buffer).unwrap()
    }

    /// Predicts the votes of a user for all articles and reports the `top` predictions to the console.
    pub fn predict_for_user(&self, name: &str, top: usize) {
        let user_id = if let Some(user_id) = self.database.users.get(name) {
            user_id
        } else {
            println!("User {} not found.", name);
            return;
        };

        let user_factor = self.user_factors.row(*user_id);

        let mut predictions = Vec::new();
        for (article, article_id) in self.database.articles.iter() {
            let article_factor = self.article_factors.row(*article_id);
            let prediction = user_factor.dot(&article_factor);
            predictions.push((article, prediction));
        }

        let mut sorted_predictions = predictions
            .iter()
            .enumerate()
            .filter(|(article_id, _)| !self.user_votes[*user_id].contains(article_id))
            .map(|(_, (article, prediction))| (*article, *prediction))
            .collect::<Vec<_>>();

        sorted_predictions.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        print!("User {} will most likely upvote those articles: ", name);
        for (article, prediction) in sorted_predictions.iter().take(top) {
            print!("{} (predicted vote: {:.2}), ", article, prediction);
        }
        println!();
    }

    /// Predicts the votes of all users for a given article and reports the `top` predictions to the console.
    pub fn predict_for_article(&self, name: &str, top: usize) {
        let article_id = if let Some(article_id) = self.database.articles.get(name) {
            *article_id
        } else {
            println!("Article not found.");
            return;
        };

        let article_factor = self.article_factors.row(article_id);
        let mut predictions = Vec::new();
        for (user, user_id) in self.database.users.iter() {
            let user_factor = self.user_factors.row(*user_id);
            let prediction = user_factor.dot(&article_factor);
            predictions.push((user, prediction, user_id));
        }

        let mut sorted_predictions = predictions
            .iter()
            .filter(|(_, _, user_id)| {
                !self.database.article_votes[article_id]
                    .iter()
                    .any(|(id, _)| id == *user_id)
            })
            .map(|(user, prediction, _)| (*user, *prediction))
            .collect::<Vec<_>>();
        //
        sorted_predictions.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        print!("{} will most likely be upvoted by: ", name);
        for (user, prediction) in sorted_predictions.iter().take(top) {
            println!("{} (predicted vote: {:.2}), ", user, prediction);
        }
        println!();
    }
}
