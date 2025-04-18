# 🏏 IPL Match Winner Predictor

This project predicts the winner of an IPL (Indian Premier League) cricket match using machine learning. It analyzes past team performances and venue data to forecast match outcomes and visualize win probabilities.

---

## 📂 Project Structure

- `iplpred.py` – Main script containing all logic (data processing, model training, prediction, visualization).
- `ipl_colab.csv` – Required dataset. Make sure this file is placed at the path specified in the script (`C:/Users/Private Fox/Downloads/ipl_colab.csv`).

---

## 🚀 Features

- Predict match winners based on teams and venue.
- Uses a **Random Forest Classifier** for predictions.
- Visualizes average team scores and win probabilities.
- Interactive CLI interface.
- Easy to extend or integrate into a web app.

---

## 🧠 How It Works

1. Loads IPL data and filters for consistent teams/venues.
2. Calculates average and highest team scores.
3. Trains a Random Forest Classifier on:
   - Venue (encoded)
   - Team 1 & Team 2 (encoded)
   - Team average scores
4. Takes user input (teams and venue) and predicts:
   - Winner
   - Win probability
   - Visualization with bar charts

---

## 📦 Requirements

Install dependencies using pip:

```bash
pip install pandas numpy matplotlib scikit-learn




**🏁 How to Run**
1.Ensure the dataset (ipl_colab.csv) exists at the path used in the script.

2.Run the script:
**python iplpred.py**

3.Follow prompts to choose teams and venue.

4.View predicted winner and a visual probability plot.



**📊 Example Output**

Enter number for Team 1: 1
Enter number for Team 2: 4
Enter number for Venue: 2

Predicted Winner: Chennai Super Kings
Win Probability: 72.30%


🤝 Contributing
Contributions are welcome! Feel free to fork the repo, improve the logic, or create a web interface for it.

📄 License
This project is licensed under the MIT License.
