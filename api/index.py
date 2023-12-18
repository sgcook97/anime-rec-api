import json
import os
from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
from dotenv import load_dotenv, find_dotenv
from psycopg2 import DataError, IntegrityError
import requests
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timedelta, timezone
from flask_jwt_extended import create_access_token,get_jwt,get_jwt_identity, \
                               unset_jwt_cookies, jwt_required, JWTManager
from werkzeug.security import generate_password_hash, check_password_hash
from keras.models import Model
from keras.layers import Input, Embedding, Flatten, Dense, Concatenate
from keras.optimizers import Adam
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import process

#app instance
load_dotenv(find_dotenv())
app = Flask(__name__)
CORS(app)

# jwt config/init
app.config["JWT_SECRET_KEY"] = os.getenv("JWT_SECRET_KEY")
app.config["JWT_ACCESS_TOKEN_EXPIRES"] = timedelta(hours=1)
jwt = JWTManager(app)

# db config/init
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv("DATABASE_URL")
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)


####################################################################
####################################################################


"""
    Getting models ready
"""

########################
# NEURAL NETWORK MODEL #
########################

nn_model = None
training_model = None

class NeuralNetworkRecommender:
    def __init__(self, num_users, num_items, embedding_size=50):
        """
        Initialize NeuralNetworkRecommender with user and item embedding configurations.

        Args:
        - num_users (int): Number of users.
        - num_items (int): Number of items.
        - embedding_size (int, optional): Size of the embedding vectors. Defaults to 50.
        """
        # Define the architecture of the neural network model
        user_input = Input(shape=[1])
        item_input = Input(shape=[1])

        # Create user and item embeddings
        user_embedding = Embedding(input_dim=num_users+1, output_dim=embedding_size)(user_input)
        item_embedding = Embedding(input_dim=num_items+1, output_dim=embedding_size)(item_input)

        # Flatten embeddings to vectors
        user_vecs = Flatten()(user_embedding)
        item_vecs = Flatten()(item_embedding)

        # Concatenate user and item vectors
        concatenated = Concatenate()([user_vecs, item_vecs])

        # Dense layers for learning interactions between user and item embeddings
        dense1 = Dense(64, activation='relu')(concatenated)
        
        # Output layer
        output = Dense(1)(dense1)

        # Create the model
        self.model = Model(inputs=[user_input, item_input], outputs=output)
        self.model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.001))
        self.ratings = None
        self.items = None

    def set_data(self, items, ratings, user_id_map, anime_id_map, orig_user_id_map, orig_anime_id_map):
        """
        Set data internally for the recommender system.

        Args:
        - items: Items data.
        - ratings: Ratings data.
        - user_id_map: User ID mapping.
        - anime_id_map: Anime ID mapping.
        - orig_user_id_map: Original user ID mapping.
        - orig_anime_id_map: Original anime ID mapping.
        """
        # Set the ratings and items data internally
        self.ratings = ratings
        self.items = items
        self.user_id_map = user_id_map
        self.anime_id_map = anime_id_map
        self.orig_user_id_map = orig_user_id_map
        self.orig_anime_id_map = orig_anime_id_map

    def train(self, user_item_pairs, ratings, num_epochs=10, batch_size=64):
        """
        Train the neural network model using user-item pairs and ratings.

        Args:
        - user_item_pairs: Pairs of users and items.
        - ratings: Ratings associated with user-item pairs.
        - num_epochs (int, optional): Number of training epochs. Defaults to 10.
        - batch_size (int, optional): Batch size for training. Defaults to 64.
        """
        # Extract user and item arrays from user-item pairs
        user_array, item_array = zip(*user_item_pairs)
        user_array = np.array(user_array)
        item_array = np.array(item_array)
        ratings = np.array(ratings)

        # Fit the model using the training data
        self.model.fit([user_array, item_array], ratings, epochs=num_epochs, batch_size=batch_size)

    def predict_ratings_for_user(self, user_id, item_ids):
        """
        Predict ratings for a user given a list of items.

        Args:
        - user_id (int): User ID.
        - item_ids (list): List of item IDs for which predictions are to be made.

        Returns:
        - dict: Dictionary with item IDs as keys and predicted ratings as values.
        """
        # Generate predictions for a user and a list of items
        user_array = np.full(len(item_ids), user_id)
        predictions = self.model.predict([user_array, np.array(item_ids)])
        predicted_ratings = {item_id: rating for item_id, rating in zip(item_ids, predictions.flatten())}
        return predicted_ratings
    
    def getTopN(self, user_id, n_recs=10):
        """
        Get top N recommendations for a given user.

        Args:
            user_id (int): The ID of the user to retrieve reccomendations for.
            n_recs (int, optional): Number of recommendations to retrieve. Default is 10.

        Raises:
            ValueError: If the data (items and ratings) has not been set using `set_data(items, ratings)`.

        Returns:
            list: A list of tuples (item_id, predicted_rating), containing top N recommendations for the user 
                sorted by predicted rating in descending order.
        """
        if self.ratings is None or self.items is None:
            raise ValueError("Data has not been set. Call set_data(items, ratings) first.")

        user_rated_items = self.ratings[self.ratings['userId'] == user_id]['animeId'].tolist()
        unrated_items = self.items[~self.items['mapped_animeId'].isin(user_rated_items)]['mapped_animeId'].tolist()
        all_predictions = self.predict_ratings_for_user(user_id, unrated_items)
        top_n_recommendations = sorted(all_predictions.items(), key=lambda x: x[1], reverse=True)[:n_recs]

        return top_n_recommendations


#######################
# CONTENT BASED MODEL #
#######################

cb_model = None

class ContentBasedRecommender:
    def __init__(self):
        """
        Initialize ContentBasedRecommender
        """
        self.sim_matrix = None
        self.anime_data = None
        self.anime_id_map = None
        self.reverse_anime_id_map = None

    def train(self, anime_features):
        """
        Train the content-based recommender system.

        Args:
        - anime_features: Features of the anime for similarity calculation.
        """
        self.sim_matrix = cosine_similarity(anime_features)

    def set_data(self, anime_features, anime_id_map, reverse_anime_id_map):
        """
        Set data for the content-based recommender system.

        Args:
        - anime_features: Features of the anime.
        - anime_id_map: Mapping of anime IDs.
        - reverse_anime_id_map: Reverse mapping of anime IDs.
        """
        self.anime_data = anime_features
        self.anime_id_map = anime_id_map
        self.reverse_anime_id_map = reverse_anime_id_map

    def getTopNItemCB(self, animeId, num_recs=10):
        """
        Get top N item recommendations based on a given anime ID.

        Args:
        - animeId: ID of the anime for which recommendations are sought.
        - num_recs (int, optional): Number of recommendations to return. Defaults to 10.

        Returns:
        - list: List of anime IDs recommended for the given anime ID.
        """
        idx = self.anime_id_map[animeId]
        sim_scores = list(enumerate(self.sim_matrix[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        anime_ids = [i[0] for i in sim_scores[1:num_recs+1]]
        return anime_ids

    def getTopNUserCB(self, user_watched_anime_ids, top_n=10):
        """
        Get top N user-based recommendations based on watched anime IDs.

        Args:
        - user_watched_anime_ids (list): List of anime IDs watched by the user.
        - top_n (int, optional): Number of recommendations to return. Defaults to 10.

        Returns:
        - list: List of recommended anime IDs for the user.
        """
        anime_to_compare = self.anime_data.drop(user_watched_anime_ids, errors='ignore')

        if anime_to_compare.empty:
            return []

        user_anime_data = self.anime_data.loc[user_watched_anime_ids]
        user_profile = user_anime_data.mean(axis=0)

        similarity_scores = cosine_similarity([user_profile], anime_to_compare)
        similar_anime_indices = similarity_scores.argsort()[0][-top_n:][::-1]
        similar_anime_ids = anime_to_compare.index[similar_anime_indices].tolist()

        return similar_anime_ids

####################################################################
####################################################################


"""
    Database Stuff
"""

class Anime(db.Model):
    animeId = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(100), nullable=False)
    avgRating = db.Column(db.Float(4,2), nullable=False)
    weightedAvg = db.Column(db.Float(4,2), nullable=False)

    def __repr__(self):
        return f"Anime(title={self.title}, avgRating={self.avgRating}, weightedAvg={self.weightedAvg})"
    
    def __init__(self, animeId, title, avgRating, weightedRating):
        self.animeId = animeId
        self.title = title
        self.avgRating = avgRating
        self.weightedAvg = weightedRating


class AnimeFeatures(db.Model):
    animeId = db.Column(db.Integer, primary_key=True)
    action = db.Column(db.Integer, default=0)
    adventure = db.Column(db.Integer, default=0)
    comedy = db.Column(db.Integer, default=0)
    drama = db.Column(db.Integer, default=0)
    sciFi = db.Column(db.Integer, default=0)
    space = db.Column(db.Integer, default=0)
    mystery = db.Column(db.Integer, default=0)
    magic = db.Column(db.Integer, default=0)
    police = db.Column(db.Integer, default=0)
    supernatural = db.Column(db.Integer, default=0)
    fantasy = db.Column(db.Integer, default=0)
    shounen = db.Column(db.Integer, default=0)
    sports = db.Column(db.Integer, default=0)
    josei = db.Column(db.Integer, default=0)
    romance = db.Column(db.Integer, default=0)
    sliceofLife = db.Column(db.Integer, default=0)
    cars = db.Column(db.Integer, default=0)
    seinen = db.Column(db.Integer, default=0)
    horror = db.Column(db.Integer, default=0)
    psychological = db.Column(db.Integer, default=0)
    thriller = db.Column(db.Integer, default=0)
    martialArts = db.Column(db.Integer, default=0)
    superPower = db.Column(db.Integer, default=0)
    school = db.Column(db.Integer, default=0)
    ecchi = db.Column(db.Integer, default=0)
    vampire = db.Column(db.Integer, default=0)
    historical = db.Column(db.Integer, default=0)
    military = db.Column(db.Integer, default=0)
    dementia = db.Column(db.Integer, default=0)
    mecha = db.Column(db.Integer, default=0)
    demons = db.Column(db.Integer, default=0)
    samurai = db.Column(db.Integer, default=0)
    game = db.Column(db.Integer, default=0)
    shoujo = db.Column(db.Integer, default=0)
    harem = db.Column(db.Integer, default=0)
    music = db.Column(db.Integer, default=0)
    shoujoAi = db.Column(db.Integer, default=0)
    shounenAi = db.Column(db.Integer, default=0)
    kids = db.Column(db.Integer, default=0)
    hentai = db.Column(db.Integer, default=0)
    parody = db.Column(db.Integer, default=0)
    yuri = db.Column(db.Integer, default=0)
    yaoi = db.Column(db.Integer, default=0)
    tv = db.Column(db.Integer, default=0)
    movie = db.Column(db.Integer, default=0)
    ova = db.Column(db.Integer, default=0)
    special = db.Column(db.Integer, default=0)
    ona = db.Column(db.Integer, default=0)

    def __init__(self, animeId, action, adventure, comedy, drama, scifi, space, mystery, magic, police, supernatural,
                 fantasy, shounen, sports, josei, romance, slice_of_life, cars, seinen, horror, psychological, thriller,
                 martial_arts, super_power, school, ecchi, vampire, historical, military, dementia, mecha, demons, samurai,
                 game, shoujo, harem, music, shoujo_ai, shounen_ai, kids, hentai, parody, yuri, yaoi, tv, movie, ova,
                 special, ona):
        self.animeId = animeId
        self.action = action
        self.adventure = adventure
        self.comedy = comedy
        self.drama = drama
        self.scifi = scifi
        self.space = space
        self.mystery = mystery
        self.magic = magic
        self.police = police
        self.supernatural = supernatural
        self.fantasy = fantasy
        self.shounen = shounen
        self.sports = sports
        self.josei = josei
        self.romance = romance
        self.slice_of_life = slice_of_life
        self.cars = cars
        self.seinen = seinen
        self.horror = horror
        self.psychological = psychological
        self.thriller = thriller
        self.martial_arts = martial_arts
        self.super_power = super_power
        self.school = school
        self.ecchi = ecchi
        self.vampire = vampire
        self.historical = historical
        self.military = military
        self.dementia = dementia
        self.mecha = mecha
        self.demons = demons
        self.samurai = samurai
        self.game = game
        self.shoujo = shoujo
        self.harem = harem
        self.music = music
        self.shoujo_ai = shoujo_ai
        self.shounen_ai = shounen_ai
        self.kids = kids
        self.hentai = hentai
        self.parody = parody
        self.yuri = yuri
        self.yaoi = yaoi
        self.tv = tv
        self.movie = movie
        self.ova = ova
        self.special = special
        self.ona = ona

    def __repr__(self):
        return f'<AnimeFeatures animeId={self.animeId}>'


class User(db.Model):
    username = db.Column(db.String(100), primary_key=True)
    password = db.Column(db.String(255), nullable=False)
    userId = db.Column(db.Integer, nullable=False)
    hasRatings = db.Column(db.Boolean, default=False)

    def __repr__(self):
        return f"User(userId={self.userId}, username={self.username})"
    
    def __init__(self, userId, username, password, hasRatings):
        self.userId = userId
        self.username = username
        self.password = password
        self.hasRatings = hasRatings


class Ratings(db.Model):
    userId = db.Column(db.Integer, index=True)
    animeId = db.Column(db.Integer, index=True)
    rating = db.Column(db.Float(4,3), nullable=False)

    __table_args__ = (
        db.PrimaryKeyConstraint(
            userId, animeId
        ),
    )

    def __repr__(self):
        return f"Rating(userId={self.userId}, animeId={self.animeId}, rating={self.rating})"
    
    def __init__(self, userId, animeId, rating):
        self.userId = userId
        self.animeId = animeId
        self.rating = rating

with app.app_context():
    db.create_all()

####################################################################
####################################################################


"""
    MyAnimeList api stuff
"""


MAL_API_URL = "https://api.myanimelist.net/v2/anime"

def getDetails(animeId):
     MAL_SEARCH_PATH = MAL_API_URL + f"/{animeId}" 
     details = requests.get(MAL_SEARCH_PATH, headers=
                            {'X-MAL-CLIENT-ID' : os.getenv("CLIENT_ID")})
     return details.json()


####################################################################
####################################################################


"""
    Helper Functions
"""

def anime_finder(title):
    """
    Find anime titles similar to the given title.

    Args:
    - title (str): Title to search for similarity.

    Returns:
    - list: List of dictionaries containing similar anime titles and their IDs.
    """
    all_anime = Anime.query.with_entities(Anime.animeId, Anime.title).all()

    all_titles = [anime.title for anime in all_anime]
    all_ids = [anime.animeId for anime in all_anime]

    closest_match = process.extract(title, all_titles, limit=15)

    closest_match_res = []

    for match in closest_match:
        closest_match_title = match[0]
        closest_match_id = all_ids[all_titles.index(closest_match_title)]

        closest_match_res.append({
            'title': closest_match_title,
            'animeId': closest_match_id 
        })

    return closest_match_res

def createAnimeDf():
    """
    Create a DataFrame containing anime data.

    Returns:
    - pandas.DataFrame: DataFrame containing anime data.
    """
    engine = db.engine
    sql_query = "SELECT * FROM anime"
    df = pd.read_sql_query(sql_query, engine)
    return df

def createRatingsDf():
    """
    Create a DataFrame containing ratings data.

    Returns:
    - pandas.DataFrame: DataFrame containing ratings data.
    """
    engine = db.engine
    sql_query = "SELECT * FROM ratings"
    df = pd.read_sql_query(sql_query, engine)
    return df

def createAnimeFeaturesDf():
    """
    Create a DataFrame containing anime features.

    Returns:
    - pandas.DataFrame: DataFrame containing anime features.
    """
    anime_features_query = 'SELECT * FROM anime_features'
    anime_features = pd.read_sql(anime_features_query, db.engine)
    return anime_features

def getNumUsers():
    """
    Get the number of unique users.

    Returns:
    - int: Number of unique users.
    """
    num_users = db.session.query(Ratings).distinct(Ratings.userId).count()
    return num_users

def getNumItems():
    """
    Get the number of unique anime items.

    Returns:
    - int: Number of unique anime items.
    """
    num_items = db.session.query(Anime).distinct(Anime.animeId).count()
    return num_items

def getTrainingData(ratings_df):
    """
    Extract user-item pairs and ratings from ratings DataFrame.

    Args:
    - ratings_df (pandas.DataFrame): DataFrame containing ratings data.

    Returns:
    - tuple: Tuple containing user-item pairs list and ratings list.
    """
    user_item_pairs = ratings_df[['userId', 'animeId']].values.tolist()
    ratings = ratings_df['rating'].values.tolist()
    return user_item_pairs, ratings

def getTopAnime(num_anime):
    """
    Get top N anime based on weighted average.

    Args:
    - num_anime (int): Number of top anime to retrieve.

    Returns:
    - list: List of dictionaries containing details of top anime.
    """
    top_anime = Anime.query.order_by(Anime.weightedAvg.desc()).limit(num_anime).all()
    data = []
    for anime in top_anime:
        details = getDetails(anime.animeId)
        if details.get('synopsis'):
            synopsis = details['synopsis']
        else:
            synopsis = 'A very good and entertaining anime (synopsis not available)'

        if details.get('main_picture'):
            data.append({
                'animeId': int(anime.animeId),
                'title': details['title'],
                'pic': details['main_picture']['medium'],
                'synopsis': synopsis
            })
        else:
            data.append({
                'animeId': int(anime.animeId),
                'title': details['title'],
                'synopsis': synopsis
            })
    return data

def getUserWatchedIds(userId):
    """
    Get the list of anime IDs watched by a user.

    Args:
    - userId (int): ID of the user.

    Returns:
    - list: List of anime IDs watched by the user.
    """
    user_watched_anime_ids = db.session.query(Ratings.animeId).filter(Ratings.userId == userId).all()
    user_watched_anime_ids = [anime_id for (anime_id,) in user_watched_anime_ids]
    return user_watched_anime_ids

def retrain_model():
    """
    Retrain the recommender model using updated data.

    Returns:
    - NeuralNetworkRecommender: Newly trained recommender model.
    """
    anime_df = createAnimeDf()
    ratings_df = createRatingsDf()

    user_id_map = {orig: mapped_id for mapped_id, orig in enumerate(ratings_df['userId'].unique(), start=1)}
    anime_id_map = {orig: mapped_id for mapped_id, orig in enumerate(ratings_df['animeId'].unique(), start=1)}

    ratings_df['origUserId'] = ratings_df['userId']
    ratings_df['origAnimeId'] = ratings_df['animeId']

    ratings_df['userId'] = ratings_df['userId'].map(user_id_map)
    ratings_df['animeId'] = ratings_df['animeId'].map(anime_id_map)

    anime_df['mapped_animeId'] = anime_df['animeId'].map(anime_id_map)

    orig_user_id_map = {orig: mapped_id for mapped_id, orig in enumerate(ratings_df['userId'].unique())}
    orig_anime_id_map = {orig: mapped_id for mapped_id, orig in enumerate(ratings_df['animeId'].unique())}

    num_users = ratings_df['userId'].nunique()
    num_items = ratings_df['animeId'].nunique()

    new_nn_model = NeuralNetworkRecommender(num_users, num_items)

    new_nn_model.set_data(anime_df, ratings_df, user_id_map, anime_id_map, orig_user_id_map, orig_anime_id_map)

    user_item_pairs, ratings = getTrainingData(ratings_df)
    
    new_nn_model.train(user_item_pairs, ratings) 

    return new_nn_model


####################################################################
####################################################################


"""
    jwt login Stuff
"""

# refresh token
@app.after_request
def refresh_expiring_jwts(response):
    """
    Refresh expiring JWTs in the response.

    Args:
    - response: HTTP response object.

    Returns:
    - HTTP response: Updated response object with refreshed access token.
    """
    try:
        exp_timestamp = get_jwt()["exp"]
        now = datetime.now(timezone.utc)
        target_timestamp = datetime.timestamp(now + timedelta(minutes=30))
        if target_timestamp > exp_timestamp:
            access_token = create_access_token(identity=get_jwt_identity())
            data = response.get_json()
            if type(data) is dict:
                data["access_token"] = access_token 
                response.data = json.dumps(data)
        return response
    except (RuntimeError, KeyError):
        return response

# logout
@app.route("/api/logout", methods=["POST"])
def logout():
    """
    Logout the user by removing JWT cookies.

    Returns:
    - flask.Response: JSON response confirming successful logout.
    """
    response = jsonify({"msg": "logout successful"})
    unset_jwt_cookies(response)
    return response


# register user
@app.route("/api/register", methods=["POST", "OPTIONS"])
def register():
    """
    Register a new user.

    Returns:
    - flask.Response: JSON response indicating the success or failure of user registration.
    """
    if request.method == "OPTIONS":
        response = jsonify({'Allow': 'POST'})
        response.headers.add('Access-Control-Allow-Methods', 'POST')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type, Authorization')
        return response
    
    username = request.json.get("username", None)
    password = request.json.get("password", None)
    hasRatings = request.json.get("hasRatings", False)

    if not username or not password:
        return jsonify({'msg': 'Missing username or password'}), 400

    existing_user = User.query.filter_by(username=username).first()
    if existing_user:
        return jsonify({'message': f'User with username "{username}" already exists'}), 400
    
    try:
        # Get the next userId
        max_user_id = Ratings.query.order_by(Ratings.userId.desc()).first()
        if max_user_id:
            next_user_id = max_user_id.userId + 1
        else:
            next_user_id = 1

        hashed_password = generate_password_hash(password)
        new_user = User(username=username, password=hashed_password, userId=next_user_id, hasRatings=hasRatings)
        db.session.add(new_user)
        db.session.commit()

        access_token = create_access_token(identity=next_user_id)
        userData = {
            'userId': new_user.userId,
            'username': new_user.username,
            'hasRatings': new_user.hasRatings
        }
        return jsonify({'message': 'User registered successfully', 'access_token': access_token, 'userData': userData}), 201
    except IntegrityError as e:
        db.session.rollback()
        return jsonify({'message': f'Error registering user: {str(e)}'}), 500
    except DataError as e:
        db.session.rollback()
        return jsonify({'message': f'Error registering user: {str(e)}'}), 500
    except Exception as e:
        db.session.rollback()
        return jsonify({'message': f'Error registering user: {str(e)}'}), 500


# user login
@app.route("/api/login", methods=["POST", "OPTIONS"])
def login():
    """
    Authenticate user credentials and generate access tokens.

    Returns:
    - flask.Response: JSON response containing access token and user data upon successful login.
    """
    if request.method == "OPTIONS":
        response = jsonify({'Allow': 'POST'})
        response.headers.add('Access-Control-Allow-Methods', 'POST')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type, Authorization')
        return response
    
    username = request.json.get("username", None)
    password = request.json.get("password", None)

    if not username or not password:
        return jsonify({'msg': 'Missing username or password'}), 400

    user = User.query.filter_by(username=username).first()
    if not user or not check_password_hash(user.password, password):
        return jsonify({'msg': 'Invalid username or password'}), 401

    access_token = create_access_token(identity=user.userId)
    userData = {
        'userId' : user.userId,
        'username' : user.username,
        'hasRatings' : user.hasRatings
    }
    return jsonify({'access_token': access_token, 'userData': userData}), 200


####################################################################
####################################################################


"""
    test api
"""

##############################
##############################
##                          ##
##   FOR TESTING PURPOSES   ##
##                          ##
##############################
##############################
@app.route("/api/test", methods=["GET"])
def test_api():
    anime_df = createAnimeDf()
    ratings_df = createRatingsDf()

    user_id_map = {int(orig): int(mapped_id) for mapped_id, orig in enumerate(ratings_df['userId'].unique(), start=1)}
    anime_id_map = {int(orig): int(mapped_id) for mapped_id, orig in enumerate(ratings_df['animeId'].unique(), start=1)}

    ratings_df['userId'] = ratings_df['userId'].map(user_id_map)
    ratings_df['animeId'] = ratings_df['animeId'].map(anime_id_map)

    anime_df['mapped_animeId'] = anime_df['animeId'].map(anime_id_map)

    orig_user_id_map = {mapped_id: orig for orig, mapped_id in user_id_map.items()}
    orig_anime_id_map = {mapped_id: orig for orig, mapped_id in anime_id_map.items()}

    num_users = ratings_df['userId'].nunique()
    num_items = ratings_df['animeId'].nunique()

    len_id_map = len(user_id_map)
    len_orig_map = len(orig_user_id_map)

    num = orig_user_id_map[15]

    item_id = np.random.choice(anime_df['mapped_animeId'].values)
    orig_id = orig_anime_id_map[item_id]
    title = anime_df[anime_df['animeId'] == orig_id]['title'].values[0]

    data = {
        'num_items' : num_items,
        'num_users' : num_users,
        'id' : num,
        'map_len' : len_id_map,
        'orig_map_len' : len_orig_map,
        'title' : title
    }
    return jsonify(data)

####################################################################
####################################################################

"""
    model api stuff
"""
# used to initialize the model
@app.route("/api/initialize", methods=["GET"])
def initialize_recommender():
    """
    Initialize the recommender system by creating and training neural network and content-based models.

    Returns:
    - flask.Response: JSON response indicating successful initialization of the recommender.
    """
    global nn_model
    global cb_model

    # get data
    anime_df = createAnimeDf()
    ratings_df = createRatingsDf()
    anime_features_df = createAnimeFeaturesDf()

    # nueral network
    user_id_map = {orig: mapped_id for mapped_id, orig in enumerate(ratings_df['userId'].unique(), start=1)}
    anime_id_map = {orig: mapped_id for mapped_id, orig in enumerate(ratings_df['animeId'].unique(), start=1)}

    ratings_df['origUserId'] = ratings_df['userId']
    ratings_df['origAnimeId'] = ratings_df['animeId']

    ratings_df['userId'] = ratings_df['userId'].map(user_id_map)
    ratings_df['animeId'] = ratings_df['animeId'].map(anime_id_map)

    anime_df['mapped_animeId'] = anime_df['animeId'].map(anime_id_map)

    orig_user_id_map = {v: k for k, v in user_id_map.items()}
    orig_anime_id_map = {v: k for k, v in anime_id_map.items()}

    num_users = ratings_df['userId'].nunique()
    num_items = ratings_df['animeId'].nunique()
    
    nn_model = NeuralNetworkRecommender(num_users, num_items)
    
    nn_model.set_data(anime_df, ratings_df, user_id_map, anime_id_map, orig_user_id_map, orig_anime_id_map)
    
    user_item_pairs, ratings = getTrainingData(ratings_df)
    nn_model.train(user_item_pairs, ratings)
    
    
    # content based
    cb_anime_id_map = {orig: mapped_id for mapped_id, orig in enumerate(anime_features_df['animeId'].unique(), start=0)}
    cb_reverse_anime_id_map = {v: k for k, v in cb_anime_id_map.items()}

    cb_model = ContentBasedRecommender()

    anime_features_df = anime_features_df.set_index('animeId')
    
    cb_model.set_data(anime_features_df, cb_anime_id_map, cb_reverse_anime_id_map)
    
    cb_model.train(anime_features_df)


    return jsonify({"message": "Recommender initialized"})


@app.route("/api/update-model", methods=["POST"])
def update_model():
    """
    Update the model by retraining it and replacing the global model with the retrained one.

    Returns:
    - flask.Response: JSON response indicating successful completion of the model update.
    """
    global training_model
    global nn_model

    training_model = retrain_model()
    nn_model = training_model
    training_model = None

    return jsonify({"message": "Model update completed"})


####################################################################
####################################################################


"""
    api stuff
"""


# /api/recs
@app.route("/api/recs", methods=["GET"])
@jwt_required()
def return_recs():
    """
    Get anime recommendations based on the user's watched history.

    Returns:
    - flask.Response: JSON response containing recommended anime details.
    """

    orig_user_id = int(request.args.get('userId'))
    numRecs = 15
    detailed_recs = []


    # If the user has been trained into the NN model, we will use that
    if orig_user_id in nn_model.user_id_map.keys():
        print('here')
        userId = nn_model.user_id_map[orig_user_id]
        recommendations = nn_model.getTopN(userId, numRecs)
        for item_id, _ in recommendations:
            orig_id = int(nn_model.orig_anime_id_map[item_id])
            details = getDetails(orig_id)
            if details.get('synopsis'):
                synopsis = details['synopsis']
            else:
                synopsis = 'A very good and entertaining anime (synopsis not available)'

            if details.get('main_picture'):
                pic_url = details['main_picture']['medium']
                anime_detail = {
                    'animeId': orig_id,
                    'title': details['title'],
                    'pic': pic_url,
                    'synopsis': synopsis
                }
            elif details.get('title'):
                anime_detail = {
                    'animeId': orig_id,
                    'title': details['title'],
                    'synopsis': synopsis
                }
            else:
                anime_detail = {
                    'animeId': orig_id,
                    'title' : 'not found',
                    'synopsis': synopsis
                }
            detailed_recs.append(anime_detail)
    # If the user is new, we will use the content based recommender
    else:
        user_watched_ids = getUserWatchedIds(orig_user_id)
        recommendations = cb_model.getTopNUserCB(user_watched_ids, numRecs)
        for item_id in recommendations:
            orig_id = int(cb_model.reverse_anime_id_map[item_id])
            details = getDetails(orig_id)
            if details.get('synopsis'):
                synopsis = details['synopsis']
            else:
                synopsis = 'A very good and entertaining anime (synopsis not available)'

            if details.get('main_picture'):
                pic_url = details['main_picture']['medium']
                anime_detail = {
                    'animeId': orig_id,
                    'title': details['title'],
                    'pic': pic_url,
                    'synopsis': synopsis
                }
            elif details.get('title'):
                anime_detail = {
                    'animeId': orig_id,
                    'title': details['title'],
                    'synopsis': synopsis
                }
            else:
                anime_detail = {
                    'animeId': orig_id,
                    'title' : 'not found',
                    'synopsis': synopsis
                }
            detailed_recs.append(anime_detail)

    return jsonify(userId=orig_user_id, recs=detailed_recs), 200


@app.route("/api/topanime", methods=['GET'])
def return_top_anime():
    """
    Get top anime recommendations.

    Returns:
    - flask.Response: JSON response containing details of top anime.
    """
    num_anime = request.args.get('num_anime', default=15, type=int)
    top_anime_data = getTopAnime(num_anime)
    return jsonify(top_anime_data), 200


@app.route("/api/submit-ratings", methods=['POST'])
@jwt_required()
def submit_ratings():
    """
    Submit user ratings for anime.

    Returns:
    - flask.Response: JSON response indicating successful submission of ratings.
    """
    current_user_id = int(get_jwt_identity())
    user_id = int(request.json.get('userId'))
    anime_ids = request.json.get('animeIds')

    if current_user_id != user_id:
        return jsonify({'message': 'Unauthorized'}), 401

    try:
        user = User.query.filter_by(userId=user_id).first()
        user.hasRatings = True;
        userData = {
            'userId' : user.userId,
            'username' : user.username,
            'hasRatings' : user.hasRatings
        }
        db.session.commit()
        for anime_id in anime_ids:
            anime = Anime.query.filter_by(animeId=anime_id).first()
            if anime:
                rating = anime.weightedAvg
                new_rating = Ratings(userId=user_id, animeId=anime_id, rating=rating)
                db.session.add(new_rating)
        db.session.commit()
        return jsonify({'message': 'Ratings submitted successfully', 'userData': userData}), 201
    except Exception as e:
        db.session.rollback()
        return jsonify({'message': 'Failed to submit ratings', 'error': str(e)}), 500

@app.route("/api/search", methods=["GET"])
@jwt_required()
def search_anime():
    """
    Search for anime based on a provided title.

    Returns:
    - flask.Response: JSON response containing details of the found anime.
    """
    raw_title = request.args.get('title')
    anime_infos = anime_finder(raw_title)

    anime_details = []
    for anime_info in anime_infos:
        details = getDetails(int(anime_info['animeId']))

        if details.get('synopsis'):
            synopsis = details['synopsis']
        else:
            synopsis = 'A very good and entertaining anime (synopsis not available)'

        if details.get('main_picture'):
            pic_url = details['main_picture']['medium']
            anime_detail = {
                'animeId': anime_info['animeId'],
                'title': details['title'],
                'pic': pic_url,
                'synopsis': synopsis
            }
        elif details.get('title'):
            anime_detail = {
                'animeId': anime_info['animeId'],
                'title': details['title'],
                'synopsis': synopsis
            }
        else:
            anime_detail = {
                'animeId': anime_info['animeId'],
                'title' : anime_info['title'],
                'synopsis': synopsis
            }

        anime_details.append(anime_detail)

    return jsonify(anime_details), 200


@app.route('/api/content-recs', methods=["GET"])
@jwt_required()
def contentRecs():
    """
    Get content-based recommendations for a specific anime.

    Returns:
    - flask.Response: JSON response containing content-based anime recommendations.
    """
    animeId = request.args.get('animeId')
    rec_ids = cb_model.getTopNItemCB(int(animeId), 10)

    anime_details = []
    for idx in rec_ids:
        orig_id = int(cb_model.reverse_anime_id_map[idx])
        details = getDetails(orig_id)

        if details.get('synopsis'):
            synopsis = details['synopsis']
        else:
            synopsis = 'A very good and entertaining anime (synopsis not available)'

        if details.get('main_picture'):
            pic_url = details['main_picture']['medium']
            anime_detail = {
                'animeId': orig_id,
                'title': details['title'],
                'pic': pic_url,
                'synopsis': synopsis
            }
        elif details.get('title'):
            anime_detail = {
                'animeId': orig_id,
                'title': details['title'],
                'synopsis': synopsis 
            }
        else:
            anime_detail = {
                'animeId': orig_id,
                'title' : 'not found',
                'synopsis': synopsis 
            }

        anime_details.append(anime_detail)
    
    return jsonify({'animeId': animeId, 'anime_details': anime_details}), 200

@app.route('/api/rate-anime', methods=['POST'])
@jwt_required()
def rate_anime():
    """
    Rate an anime.

    Returns:
    - flask.Response: JSON response confirming successful submission of the rating.
    """
    data = request.json

    user_id = int(data.get('userId'))
    anime_id = int(data.get('animeId'))
    new_rating = int(data.get('rating'))

    rating = Ratings.query.filter_by(userId=user_id, animeId=anime_id).first()

    if rating:
        # Update the existing rating
        rating.rating = new_rating
    else:
        # Create a new rating entry
        new_rating_entry = Ratings(userId=user_id, animeId=anime_id, rating=new_rating)
        db.session.add(new_rating_entry)

    db.session.commit()
    return jsonify({'message': 'Rating submitted successfully'}), 201

@app.route("/api/healthchecker", methods=["GET"])
def healthchecker():
    """
    Check the health status of the application.

    Returns:
    - flask.Response: JSON response confirming the health status of the application.
    """
    return {"status": "success", "message": "Integrate Flask Framework with Next.js"}, 200

# if __name__ == "__main__":
#     app.run(debug=True, port=8000)