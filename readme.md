# Geyser SCP recommendation bot
A small project aimed to recreate the Funk MF [[1]](#1). rating prediction bot, originally created for the Netflix Prize competition.
But this time, it's for the SCP Wiki, specifically to recommend SCP articles to users based on their voting behavior.
I created the tool in one day as a toy project, so it's not very polished.

## Usage
The bot has a simple CLI interface. The `help` command will print a list of available commands and their usage.

With the `update` command, the bot will scrape SCP articles from the SCP Wiki and create a database of user votes for all
scraped articles.
By default, it scrapes the articles between 6000 and 7999.
This can be adjusted with the --from and --to arguments.
Note that the inhomogeneous distribution of votes for SCP articles, as well as the fluid user base,
will result in weird predictions if the scraped articles span a large time period.
```
Options:
  -f, --from [<FROM>]  The article number to start from (inclusive)
  -t, --to [<TO>]      The article number to end at (inclusive)
```

The `train` command will train the bot on the scraped data. It will use a collaborative filtering algorithm to create
a matrix decomposition akin to an [SVD](https://en.wikipedia.org/wiki/Singular_value_decomposition).
All arguments are optional, and will default to values close to those in the original Funk paper.

```
  -l, --latent_factors [<LATENT_FACTORS>]  The number of latent factors to use for the model
  -i, --iterations [<ITERATIONS>]          The number of iterations to train the model
  -r, --learning_rate [<LEARNING_RATE>]    The learning rate to use for the model
  -o, --regularization [<REGULARIZATION>]  The regularization to use for the model
```

The `predict` command will predict the rating of all articles for a user and print the top 10 recommendations.
Similarly, the `advertise` command will predict the rating of all users for an article and print the top 10 users 
(though this feature is not very useful, if you don't intend to launch a targeted advertising campaign for your article).
The range of users and articles to predict for can be adjusted with the --top argument.

```
Options:
  -t, --top [<TOP>]  The number of top articles to predict
```

## Results
I tested around with some values and left the best performing ones in the code as defaults.
The results are not very good, but I think that's mostly due to the fact that the data is not very good.
The data is very dull 
(since most users upvote more than they downvote,
there is no rating system beyond simple upvoting,
there is no way to distinguish between users who don't vote and users who haven't read the article,
and the rating is very tilted towards older articles),
so it's hard to get good results with such a simple algorithm.
If you can find better parameters, please let me know.
If you know a better way to regularize during training, please let me know.
I am very much not an expert in machine learning.

## References
<a id="1">[1]</a> Funk, Simon. "Netflix update: Try this at home." sifter.org (2006). https://sifter.org/~simon/journal/20061211.html