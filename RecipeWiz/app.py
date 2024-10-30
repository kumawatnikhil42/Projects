# app.py

from flask import Flask, render_template, request, redirect, url_for, flash, session
from db import db, User, Preferences
import pandas as pd
import requests
import os

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.secret_key = 'your_secret_key'  # Change this to a random secret key
db.init_app(app)

def get_recommendations(preferences):
    # Load recommendations from CSV
    recommendations_df = pd.read_csv('recommendations.csv')

    # Initialize user preferences
    dietary_preferences = preferences.dietary_preferences.lower() if preferences and preferences.dietary_preferences else None
    dietary_restrictions = preferences.dietary_restrictions.lower().split(',') if preferences and preferences.dietary_restrictions else []
    favorite_cuisines = preferences.favorite_cuisines.lower() if preferences and preferences.favorite_cuisines else None

    # Start filtering
    recommendations = recommendations_df.copy()

    # Filter by dietary preferences if provided
    if dietary_preferences:
        recommendations = recommendations[recommendations['dietary_preferences'].str.lower() == dietary_preferences]

    # Filter by favorite cuisines if provided
    if favorite_cuisines:
        recommendations = recommendations[recommendations['favorite_cuisines'].str.lower() == favorite_cuisines]

    # Further filter out dishes that contain any of the dietary restrictions
    if dietary_restrictions:
        recommendations['dietary_restrictions'] = recommendations['dietary_restrictions'].astype(str).fillna('')
        for restriction in dietary_restrictions:
            recommendations = recommendations[~recommendations['dietary_restrictions'].str.lower().str.contains(restriction)]

    # Remove duplicate dishes based on 'name_of_dish'
    recommendations = recommendations.drop_duplicates(subset=['name_of_dish'])

    # Return recommendations as a list of dictionaries
    return recommendations.to_dict(orient='records')


with app.app_context():
    db.create_all()  # Create database tables



@app.route('/')
def home():
    user_id = session.get('user_id')
    
    # Use db.session.get() to fetch the user
    user = db.session.get(User, user_id) if user_id else None
    username = user.full_name if user else None

    api_key = os.getenv('YOUR_YOUTUBE_API_KEY')  # Get your YouTube API key from environment variables
    user_preferences = Preferences.query.filter_by(user_id=user_id).first() if user_id else None

    # Default query if user preferences are not available
    default_query = "healthy recipes and nutritional tips"
    
    # Use dietary preferences and restrictions if available
    if user_preferences:
        dietary_preferences = user_preferences.dietary_preferences or ""
        dietary_restrictions = user_preferences.dietary_restrictions or ""
        
        # Create the final query with proper spacing
        query_parts = [part for part in [dietary_preferences, dietary_restrictions] if part]
        query = " and ".join(query_parts) + " foods ,recipes,nutritional tips" if query_parts else default_query
    else:
        query = default_query  # Use the default if no preferences found

    videos = get_youtube_videos(api_key, query)
    # print(videos)

    return render_template('home.html', username=username,videos=videos)



@app.route('/signup', methods=['POST'])
def signup():
    full_name = request.form['username']  # Use 'username' for full name
    email = request.form['email']
    password = request.form['password']

    # Check if user already exists
    if User.query.filter_by(email=email).first():
        flash('Email already exists. Please login or use another email.', 'warning')
        return redirect(url_for('login'))

    # Create new user
    role = 'superuser' if email == 'superuser@gmail.com' else 'user'  # Example condition for superuser
    new_user = User(full_name=full_name, email=email, password=password, role=role) # Hash the password in a real app
    db.session.add(new_user)
    db.session.commit()

    flash('Signup successful! Please log in.', 'success')
    return redirect(url_for('login'))

@app.route('/login', methods=['POST'])
def login():
    email = request.form['username_email']
    password = request.form['password']
    user = User.query.filter_by(email=email, password=password).first()  # Hash the password in a real app
    
    if user:
        session['user_id'] = user.id
        session['role'] = user.role  # Store user role in the session
        flash('Login successful!', 'success')
        return redirect(url_for('home'))
    else:
        flash('Login failed. Check your email and password.', 'danger')
        return redirect(url_for('login'))

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    session.pop('role', None)
    flash('You have been logged out.', 'info')
    return redirect(url_for('home'))

@app.route('/users')
def users():
    if 'role' in session and session['role'] == 'superuser':
        all_users = User.query.all()  # Fetch all users from the database
        return render_template('users.html', users=all_users)
    else:
        flash('You do not have permission to view this page.', 'danger')
        return redirect(url_for('home'))

@app.route('/delete_user/<int:user_id>', methods=['POST'])
def delete_user(user_id):
    if 'role' in session and session['role'] == 'superuser':
        user = User.query.get(user_id)  # Get the user by ID

        if user:
            db.session.delete(user)  # Delete the user from the session
            db.session.commit()  # Commit the changes
            flash('User deleted successfully.', 'success')
        else:
            flash('User not found.', 'danger')
    else:
        flash('You do not have permission to delete users.', 'danger')

    return redirect(url_for('users'))  # Redirect to the users page

@app.route('/add_user', methods=['POST'])
def add_user():
    full_name = request.form['username']  # Use 'username' for full name
    email = request.form['email']
    password = request.form['password']

    # Check if user already exists
    if User.query.filter_by(email=email).first():
        flash('Email already exists. Please use another email.', 'warning')
        return redirect(url_for('users'))

    # Create new user
      # Example condition for superuser
    new_user = User(full_name=full_name, email=email, password=password, role="superuser")  # Hash the password in a real app
    db.session.add(new_user)
    db.session.commit()

    flash('User added successfully!', 'success')
    return redirect(url_for('users'))

@app.route('/profile')
def profile():
    user_id = session.get('user_id')
    user = User.query.get(user_id) if user_id else None

    if user:
        return render_template('profile.html', user=user)
    else:
        flash('You need to log in first.', 'danger')
        return redirect(url_for('login'))
    
@app.route('/update_preferences', methods=['POST'])
def update_preferences():
    if 'user_id' not in session:
        flash('You need to log in first.')
        return redirect(url_for('login'))

    # Get the form data
    dietary_preferences = request.form['dietary_preferences']
    dietary_restrictions = request.form.getlist('dietary_restrictions')  # Get all checked values as a list
    favorite_cuisines = request.form['favorite_cuisines']
    
    user_id = session['user_id']

    # Join the dietary restrictions into a comma-separated string
    dietary_restrictions_str = ', '.join(dietary_restrictions)

    # Check if preferences already exist for the user
    preferences = Preferences.query.filter_by(user_id=user_id).first()
    if preferences:
        # Update existing preferences
        preferences.dietary_preferences = dietary_preferences
        preferences.dietary_restrictions = dietary_restrictions_str  # Save as a string
        preferences.favorite_cuisines = favorite_cuisines
    else:
        # Create new preferences entry
        preferences = Preferences(
            user_id=user_id,
            dietary_preferences=dietary_preferences,
            dietary_restrictions=dietary_restrictions_str,
            favorite_cuisines=favorite_cuisines
        )
        db.session.add(preferences)
    
    db.session.commit()  # Save changes to the database
    flash('Preferences updated successfully!')
    return redirect(url_for('profile'))

@app.route('/recommendations', methods=['GET'])
def recommendations():
    if 'user_id' not in session:
        flash('You need to log in first.', 'danger')
        return redirect(url_for('login'))  # Redirect to login page if not logged in
    
    # Assuming user is logged in
    user_preferences = Preferences.query.filter_by(user_id=session['user_id']).first()
    recommendations = []
    page = request.args.get('page', 1, type=int)  # Get the page number from the query string
    per_page = 6  # Number of recommendations per page

    if user_preferences:
        # Get recommendations based on user preferences
        recommendations = get_recommendations(user_preferences)

    # Paginate the recommendations
    total_recommendations = len(recommendations)
    start = (page - 1) * per_page
    end = start + per_page
    recommendations_page = recommendations[start:end]

    return render_template('recommendations.html', 
                           recommendations=recommendations_page, 
                           total_recommendations=total_recommendations, 
                           page=page, 
                           per_page=per_page)


import re

@app.route('/generate_recipe/<dish_name>', methods=['GET'])
def generate_recipe(dish_name):
    if 'user_id' not in session:
        flash('You need to log in first.', 'danger')
        return redirect(url_for('login'))

    # Retrieve dietary preferences and restrictions from query parameters
    dietary_preferences = request.args.get('dp', 'None')
    dietary_restrictions = request.args.get('dr', 'None')

    api_key = os.getenv('YOUR_API_KEY')

    # Modify the prompt to include dietary preferences and restrictions if relevant
    prompt = f"Generate step by step cooking recipe of {dish_name} in clear format, like Yields:,Prep Time:, Cook Time:, Ingredients:, Equipment:, Instructions:, Tips:, Enjoy!"
    
    # If dietary preferences or restrictions exist, include them in the prompt
    if dietary_preferences != 'None':
        prompt += f" Make sure the recipe aligns with the dietary preferences: {dietary_preferences}."
    if dietary_restrictions != 'None':
        prompt += f" Avoid any ingredients that conflict with these dietary restrictions: {dietary_restrictions}."
    print(prompt)
    payload = {
        "contents": [{"parts": [{"text": prompt}]}]
    }
    
    try:
        response = requests.post(
            f'https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={api_key}',
            headers={'Content-Type': 'application/json'},
            json=payload
        )

        if response.status_code == 200:
            recipe = response.json()
            generated_steps = recipe.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', 'No recipe found.')
            
            # Clean up extra symbols and format the text
            cleaned_steps = re.sub(r'(\*\*|\*|##|#)', '', generated_steps)  # Remove **, *, ##, and #
            cleaned_steps = cleaned_steps.strip()  # Remove leading/trailing whitespace

            # Process the cleaned steps into sections
            sections = {
                'yields': '',
                'prep_time': '',
                'cook_time': '',
                'ingredients': [],
                'equipment': [],
                'instructions': [],
                'tips': []
            }

            current_section = None
            for line in cleaned_steps.split('\n'):
                line = line.strip()
                if line.startswith("Yields:"):
                    sections['yields'] = line.replace("Yields:", "").strip()
                    current_section = 'yields'
                elif line.startswith("Prep Time:"):
                    sections['prep_time'] = line.replace("Prep Time:", "").strip()
                    current_section = 'prep_time'
                elif line.startswith("Cook Time:"):
                    sections['cook_time'] = line.replace("Cook Time:", "").strip()
                    current_section = 'cook_time'
                elif line.startswith("Ingredients:"):
                    current_section = 'ingredients'
                elif line.startswith("Equipment:"):
                    current_section = 'equipment'
                elif line.startswith("Instructions:"):
                    current_section = 'instructions'
                elif line.startswith("Tips:"):
                    current_section = 'tips'
                else:
                    if current_section == 'ingredients':
                        sections['ingredients'].append(line)
                    elif current_section == 'equipment':
                        sections['equipment'].append(line)
                    elif current_section == 'instructions':
                        sections['instructions'].append(line)
                    elif current_section == 'tips':
                        sections['tips'].append(line)




            return render_template('recipe.html', dish_name=dish_name, sections=sections, dietary_preferences=dietary_preferences, dietary_restrictions=dietary_restrictions)
        else:
            flash('Failed to generate recipe. Please try again.', 'danger')
            return redirect(url_for('recommendations'))
    except requests.exceptions.RequestException as e:
        flash(f'An error occurred: {str(e)}', 'danger')
        return redirect(url_for('recommendations'))

def get_youtube_videos(api_key, query, max_results=7):
    # Constructing the YouTube API URL with dynamic maxResults
    url = f"https://www.googleapis.com/youtube/v3/search?part=snippet&q={query}&key={api_key}&maxResults={max_results}&type=video"
    
    response = requests.get(url)
    videos = []

    if response.status_code == 200:
        for item in response.json().get('items', []):
            video_id = item['id']['videoId']
            video_title = item['snippet']['title']
            video_url = f"https://www.youtube.com/embed/{video_id}"  # Constructing embed URL
            
            # Check if the video is available by fetching video details
            video_details_url = f"https://www.googleapis.com/youtube/v3/videos?id={video_id}&key={api_key}&part=status"
            details_response = requests.get(video_details_url)

            # Only add the video if it is available
            if details_response.status_code == 200:
                video_status = details_response.json().get('items', [])
                if video_status and video_status[0]['status']['embeddable']:
                    videos.append({
                        'title': video_title,
                        'url': video_url,
                    })
    else:
        print(f"Error fetching videos: {response.status_code} - {response.text}")

    return videos



if __name__ == '__main__':
    from dotenv import load_dotenv
    load_dotenv() 
    app.run(debug=True)
