# db.py

from flask_sqlalchemy import SQLAlchemy

# Create an instance of SQLAlchemy
db = SQLAlchemy()

# Define the User model
class User(db.Model):
    __tablename__ = 'users'  # Table name in the database
    id = db.Column(db.Integer, primary_key=True)  # Primary key
    full_name = db.Column(db.String(100), nullable=False)  # Full name of the user
    email = db.Column(db.String(120), unique=True, nullable=False)  # Email address (unique)
    password = db.Column(db.String(120), nullable=False)  # Password
    role = db.Column(db.String(50), default='user')  # Role of the user ('user' or 'superuser')

    def __repr__(self):
        return f'<User {self.full_name}, Role: {self.role}>'

class Preferences(db.Model):
    __tablename__ = 'preferences'  # Table name in the database
    id = db.Column(db.Integer, primary_key=True)  # Primary key
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)  # Foreign key to User
    dietary_preferences = db.Column(db.String(100), nullable=True)  # Dietary preferences
    dietary_restrictions = db.Column(db.String(100), nullable=True)  # Dietary restrictions
    favorite_cuisines = db.Column(db.String(100), nullable=True)  # Favorite cuisines

    user = db.relationship('User', backref='preferences')  # Relationship with User

    def __repr__(self):
        return f'<Preferences User ID: {self.user_id}>'

class Ingredient(db.Model):
    __tablename__ = 'ingredients'  # Table name in the database
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    is_healthy = db.Column(db.Boolean, default=True)  # Indicates if the ingredient is healthy

class Recipe(db.Model):
    __tablename__ = 'recipes'  # Table name in the database
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    ingredients = db.relationship('Ingredient', secondary='recipe_ingredients')
    is_healthy = db.Column(db.Boolean, default=True)  # Indicates if the recipe is healthy

class RecipeIngredient(db.Model):
    __tablename__ = 'recipe_ingredients'  # Join table for many-to-many relationship
    recipe_id = db.Column(db.Integer, db.ForeignKey('recipes.id'), primary_key=True)
    ingredient_id = db.Column(db.Integer, db.ForeignKey('ingredients.id'), primary_key=True)

