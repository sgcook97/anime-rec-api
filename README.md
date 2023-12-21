# Samnroll API
Backend for Anime Recommender Website

**Description:** The Anime Recommender Backend is a Flask API that powers an anime recommendation system. It integrates with a PostgreSQL database and utilizes a hybrid recommendation system incorporating both content-based (cosine similarity) and collaborative filtering (TensorFlow neural network) approaches. Additionally, it connects to the MyAnimeList API for additional anime information.

## Features

- **PostgreSQL Database**: Utilizes an AWS RDS PostgreSQL database for storing anime information, user ratings, and login data.
- **Recommendation Systems**: Employs content-based and collaborative filtering recommendation models for suggesting anime to users.
- **MyAnimeList Integration**: Fetches additional anime details and image URLs from the MyAnimeList API.

## Usage
* **API Endpoints**: Explore the available API endpoints for fetching anime details, user ratings, authentication, and recommendation functionalities.
* **PostgreSQL Database**: Utilize the PostgreSQL database for storing and retrieving anime information, user ratings, and login data.

## Flask API
The Flask API is hosted on an AWS EC2 instance, ensuring it has the computational resources required for seamless interactions. It's designed to:

* Process API requests for anime details, user ratings, authentication, and personalized recommendations.
* Implement content-based and collaborative filtering recommendation models.
* Connect with the MyAnimeList API for additional anime data enrichment.

## PostgreSQL Database
The PostgreSQL database is hosted in an AWS RDS instance. The database schema includes tables for anime details, user ratings, and user authentication data. This API leverages the database to:

* Store anime data such as title, genres, average rating, etc.
* Retrieve ratings data for model training.
* Store additional anime ratings submitted by users.
* Manage user-specific data such as ratings and login credentials securely.