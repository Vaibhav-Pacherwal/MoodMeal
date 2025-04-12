import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
user_input = " ".join(sys.argv[1:])

data = pd.read_csv('emotion_dataset.csv', sep=';', names=['text', 'label'])

from sklearn.utils import resample

target_count = 400

balanced_dfs = []
for emotion in data['label'].value_counts().index:
    emotion_df = data[data['label'] == emotion]
    if len(emotion_df) > target_count:
        sampled_df = resample(emotion_df, replace=False, n_samples=target_count, random_state=42)
    else:
        sampled_df = resample(emotion_df, replace=True, n_samples=target_count, random_state=42)
    balanced_dfs.append(sampled_df)

balanced_data = pd.concat(balanced_dfs).sample(frac=1, random_state=42).reset_index(drop=True)

x = balanced_data['text']
y = balanced_data['label']


vectorizer = TfidfVectorizer()
x_vectorizer = vectorizer.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(
     x_vectorizer, y, stratify=y, test_size=0.3, random_state=42
)
model = MultinomialNB()
model.fit(x_train,y_train)

userEmo = vectorizer.transform([user_input])
predicted_mood = model.predict(userEmo)[0]

emoji_map = {
    "joy": "😊joy",
    "sadness": "😢sadness",
    "anger": "😠anger",
    "surprise": "😮surprise",
    "fear": "😨fear",
    "love": "❤️love",
    "neutral": "😐neutral"
}

predicted_mood = emoji_map.get(predicted_mood, predicted_mood)

mood_meals = {
    "😊joy": "🍓 Creamy Mac & Cheese",
    "😢sadness": "🌋 Warm Chocolate Lava Cake",
    "😠anger": "🍜 Spicy Ramen Noodles",
    "😮surprise": "🍘 Sushi (Veg/Non-veg)",
    "😨fear": "🍫 Herbal Lemon Tea with Honey Toast",
    "❤️love": "🍓 Strawberry Pancakes",
    "😐neutral": "🥪 Classic Grilled Cheese Sandwich"
}

mood_meals_recipe = {
    "😊joy": "1. Boil elbow macaroni.\n2. Prepare a cheese sauce with butter, flour, milk, and cheddar.\n3. Mix pasta and sauce, bake for 10 minutes, and serve hot!",
    "😢sadness": "1. Melt dark chocolate and butter.\n2. Whisk eggs, sugar, and flour.\n3. Bake until outer layer is firm, but center is gooey.\n4. Serve with a scoop of vanilla ice cream.",
    "😠anger": "1. Boil noodles and drain.\n2. Stir-fry with chili oil, garlic, and soy sauce.\n3. Top with a fried egg and green onions.",
    "😨fear": "1. Brew chamomile or lemon balm tea.\n2. Add lemon slices and honey.\n3. Serve with buttered toast and a peaceful playlist.",
    "😮surprise": "1. Lay out sushi rice on seaweed (nori).\n2. Add cucumber, avocado, or cooked fish.\n3. Roll tightly and slice into pieces.",
    "❤️love": "1. Prepare pancake batter with vanilla essence.\n2. Add sliced strawberries.\n3. Cook on both sides and top with whipped cream and syrup.",
    "😐neutral": "1. Butter two bread slices.\n2. Place cheddar cheese between them.\n3. Grill until golden brown and crispy.",
}

mood_meals_ytlinks = {
    "😊joy": "https://www.youtube.com/watch?v=6QXWE4QxL5Y",
    "😢sadness": "https://www.youtube.com/watch?v=vW2Z-TTnDSo",
    "😠anger": "https://www.youtube.com/watch?v=jEIjh1dJ1d8",
    "😨fear": "https://www.youtube.com/watch?v=g4GzMNEkbgQ",
    "😮surprise": "https://www.youtube.com/watch?v=I1UDS2kgqY8",
    "❤️love": "https://www.youtube.com/watch?v=0A9KDGxDZSY",
    "😐neutral": "https://www.youtube.com/watch?v=KXq4Y5dfT6k",
}

meal = mood_meals.get(predicted_mood, "🍽️ Just stay hydrated and take a deep breath!")
meal_recipe = mood_meals_recipe.get(predicted_mood)
meal_ytlink = mood_meals_ytlinks.get(predicted_mood)
print(f"{predicted_mood}::{meal}::{meal_recipe}::{meal_ytlink}")
print("Checking if data loads properly...", file=sys.stderr)

