use clap::{arg, ArgAction, command, value_parser};
use crate::database::{Database, PredictionModel};
use crate::update::Updater;

mod update;
pub(crate) mod database;

fn main() {
    let matches = command!()
        .propagate_version(true)
        .subcommand_required(true)
        .arg_required_else_help(true)
        .subcommand(
            command!("update")
                .about("Update the database by downloading articles from the wiki. Note that this will always create a new database file, overwriting any existing one.")
                .arg(arg!(-f --from [FROM] "The article number to start from (inclusive)").value_parser(value_parser!(usize)))
                .arg(arg!(-t --to [TO] "The article number to end at (inclusive)").value_parser(value_parser!(usize)))
        )
        .subcommand(
            command!("train")
                .about("train the model")
                .arg(arg!(-l --latent_factors [LATENT_FACTORS] "The number of latent factors to use for the model").value_parser(value_parser!(usize)))
                .arg(arg!(-i --iterations [ITERATIONS] "The number of iterations to train the model").value_parser(value_parser!(usize)))
                .arg(arg!(-r --learning_rate [LEARNING_RATE] "The learning rate to use for the model").value_parser(value_parser!(f64)))
                .arg(arg!(-o --regularization [REGULARIZATION] "The regularization to use for the model").value_parser(value_parser!(f64)))
        )
        .subcommand(
            command!("predict")
                .about("predict top votes on articles for a user")
                .arg(arg!(-t --top [TOP] "The number of top articles to predict").value_parser(value_parser!(usize)))
                .arg(arg!([USERS]).action(ArgAction::Append))
        )
        .subcommand(
            command!("advertise")
                .about("predict which users will most likely vote positive on an article")
                .arg(arg!(-t --top [TOP] "The number of top users to predict").value_parser(value_parser!(usize)))
                .arg(arg!([ARTICLES]).action(ArgAction::Append))
        )
        .get_matches();

    match matches.subcommand() {
        Some(("update", args)) => {
            let from = *args.get_one::<usize>("from").unwrap_or(&6000usize);
            let to = *args.get_one::<usize>("to").unwrap_or(&7999usize);
            Updater::new().update(from, to);
        },
        Some(("train", args)) => {
            let latent_factors = *args.get_one::<usize>("latent_factors").unwrap_or(&30usize);
            let iterations = *args.get_one::<usize>("iterations").unwrap_or(&120usize);
            let learning_rate = *args.get_one::<f64>("learning_rate").unwrap_or(&0.004f64);
            let regularization = *args.get_one::<f64>("regularization").unwrap_or(&0.02f64);
            let database = Database::load();
            database.train_prediction_model(latent_factors, iterations, learning_rate, regularization);
        },
        Some(("predict", args)) => {
            let prediction_model = PredictionModel::load();
            let top = args.get_one::<usize>("top").unwrap_or(&10usize);
            let users: Vec<_> = args.get_many::<String>("USERS").unwrap().collect();
            for user in users {
                prediction_model.predict_for_user(user, *top);
                println!();
            }
        }
        Some(("advertise", args)) => {
            let prediction_model = PredictionModel::load();
            let top = args.get_one::<usize>("top").unwrap_or(&10usize);
            let articles: Vec<_> = args.get_many::<String>("ARTICLES").unwrap().collect();
            for article in articles {
                prediction_model.predict_for_article(article, *top);
                println!();
            }
        }
        _ => unreachable!(),
    }
}