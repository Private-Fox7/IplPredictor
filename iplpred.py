import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Define consistent teams and venues
CONST_TEAMS = [
    'Kolkata Knight Riders', 
    'Chennai Super Kings', 
    'Rajasthan Royals',
    'Mumbai Indians', 
    'Kings XI Punjab', 
    'Royal Challengers Bangalore',
    'Delhi Daredevils', 
    'Sunrisers Hyderabad'
]

CONST_VENUES = [
    'Eden Gardens',
    'M Chinnaswamy Stadium',
    'Feroz Shah Kotla',
    'MA Chidambaram Stadium',
    'Wankhede Stadium',
    'Sawai Mansingh Stadium',
    'Punjab Cricket Association Stadium',
    'Rajiv Gandhi International Stadium'
]

def load_and_prepare_data(file_path):
    """Load and prepare IPL data for match winner prediction"""
    # Read the data
    df = pd.read_csv(file_path)
    
    # Keep only relevant features
    relevant_columns = ['batting_team', 'bowling_team', 'total', 'venue']
    df = df[relevant_columns]
    
    # Filter for consistent teams
    df = df[
        (df['batting_team'].isin(CONST_TEAMS)) & 
        (df['bowling_team'].isin(CONST_TEAMS))
    ]
    
    match_data = []
    # Group by batting team and calculate average scores
    team_stats = df.groupby('batting_team')['total'].agg(['mean', 'max']).reset_index()
    team_stats.columns = ['team', 'avg_score', 'highest_score']
    
    # Create match combinations
    for team1 in CONST_TEAMS:
        for team2 in CONST_TEAMS:
            if team1 != team2:
                team1_stats = team_stats[team_stats['team'] == team1].iloc[0]
                team2_stats = team_stats[team_stats['team'] == team2].iloc[0]
                
                # Determine winner based on average scores
                winner = team1 if team1_stats['avg_score'] > team2_stats['avg_score'] else team2
                
                for venue in CONST_VENUES:
                    match_data.append({
                        'team1': team1,
                        'team2': team2,
                        'venue': venue,
                        'team1_avg_score': team1_stats['avg_score'],
                        'team2_avg_score': team2_stats['avg_score'],
                        'winner': winner
                    })
    
    return pd.DataFrame(match_data)

def prepare_features(data):
    """Prepare features for the model"""
    # Create label encoders
    le_venue = LabelEncoder()
    le_team = LabelEncoder()
    
    # Fit team encoder on all team names
    all_teams = np.unique(data[['team1', 'team2', 'winner']].values.ravel())
    le_team.fit(all_teams)
    
    # Encode features
    data['venue_encoded'] = le_venue.fit_transform(data['venue'])
    data['team1_encoded'] = le_team.transform(data['team1'])
    data['team2_encoded'] = le_team.transform(data['team2'])
    data['winner_encoded'] = le_team.transform(data['winner'])
    
    return data, le_venue, le_team

def train_prediction_model(data):
    """Train the match winner prediction model"""
    # Prepare features
    X = data[[
        'venue_encoded', 
        'team1_encoded', 
        'team2_encoded',
        'team1_avg_score',
        'team2_avg_score'
    ]]
    y = data['winner_encoded']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Print accuracy
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {accuracy:.2f}")
    
    return model

def predict_winner(model, team1, team2, venue, le_venue, le_team, data):
    """Predict match winner based on total scores"""
    # Get team statistics
    team1_stats = data[data['team1'] == team1].iloc[0]
    team2_stats = data[data['team2'] == team2].iloc[0]
    
    # Encode inputs
    venue_encoded = le_venue.transform([venue])[0]
    team1_encoded = le_team.transform([team1])[0]
    team2_encoded = le_team.transform([team2])[0]
    
    # Prepare prediction input
    X_pred = np.array([[
        venue_encoded,
        team1_encoded,
        team2_encoded,
        team1_stats['team1_avg_score'],
        team2_stats['team2_avg_score']
    ]])
    
    # Make prediction
    proba = model.predict_proba(X_pred)[0]
    team1_prob = proba[1] * 100
    
    # Determine winner based on probability
    if team1_prob >= 50:
        winner = team1
        win_probability = team1_prob
    else:
        winner = team2
        win_probability = 100 - team1_prob
    
    return winner, win_probability, team1_prob

def plot_team_performance(data):
    """Plot average scores for all teams"""
    team_stats = data.groupby('team1')[['team1_avg_score']].mean().sort_values('team1_avg_score', ascending=False)
    
    plt.figure(figsize=(12, 6))
    plt.bar(team_stats.index, team_stats['team1_avg_score'])
    plt.xticks(rotation=45, ha='right')
    plt.title('Average Team Scores in IPL')
    plt.xlabel('Teams')
    plt.ylabel('Average Score')
    plt.tight_layout()
    plt.show()

def plot_prediction_probability(team1, team2, team1_prob):
    """Plot prediction probability for both teams"""
    teams = [team1, team2]
    probabilities = [team1_prob, 100 - team1_prob]
    
    # Set colors based on which team has higher probability
    if team1_prob >= 50:
        colors = ['#2ecc71', '#e74c3c']  # First team wins
    else:
        colors = ['#e74c3c', '#2ecc71']  # Second team wins
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(len(teams)), probabilities, color=colors)
    plt.title('Match Win Probability', fontsize=12, pad=20)
    plt.xlabel('Teams', fontsize=10)
    plt.ylabel('Win Probability (%)', fontsize=10)
    plt.ylim(0, 100)
    
    # Set team names as x-tick labels
    plt.xticks(range(len(teams)), teams, rotation=15, ha='right')
    
    # Add probability values on top of bars
    for i, (prob, bar) in enumerate(zip(probabilities, bars)):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{prob:.1f}%',
                ha='center', va='bottom')
    
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def main():
    # Load and prepare data
    print("Loading IPL data...")
    match_data = load_and_prepare_data('https://github.com/Private-Fox7/IplPredictor/blob/main/ipl_colab.csv')
    
    # Plot overall team performance
    plot_team_performance(match_data)
    
    # Prepare features and train model
    data, le_venue, le_team = prepare_features(match_data)
    model = train_prediction_model(data)
    
    while True:
        print("\nAvailable teams:")
        for i, team in enumerate(CONST_TEAMS, 1):
            print(f"{i}. {team}")
            
        # Get user input
        try:
            team1_idx = int(input("\nEnter number for Team 1: ")) - 1
            team2_idx = int(input("Enter number for Team 2: ")) - 1
            
            if team1_idx == team2_idx:
                print("Please select different teams!")
                continue
                
            print("\nAvailable venues:")
            for i, venue in enumerate(CONST_VENUES, 1):
                print(f"{i}. {venue}")
            
            venue_idx = int(input("\nEnter number for Venue: ")) - 1
            
            # Predict winner
            winner, probability, team1_prob = predict_winner(
                model, 
                CONST_TEAMS[team1_idx],
                CONST_TEAMS[team2_idx],
                CONST_VENUES[venue_idx],
                le_venue,
                le_team,
                data
            )
            
            print(f"\nPredicted Winner: {winner}")
            print(f"Win Probability: {probability:.2f}%")
            
            # Plot prediction probability
            plot_prediction_probability(
                CONST_TEAMS[team1_idx],
                CONST_TEAMS[team2_idx],
                team1_prob
            )
            
            if input("\nPredict another match? (y/n): ").lower() != 'y':
                break
                
        except (IndexError, ValueError):
            print("Invalid input! Please enter valid numbers between 1 and 8.")
            continue

if __name__ == "__main__":
    main()
