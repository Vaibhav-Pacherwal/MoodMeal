import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import sys
import io
import re

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


vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words='english',
    ngram_range=(1, 2),   
    max_df=0.9,           
    min_df=5  
)

def handle_double_negatives(text):
    double_negative_patterns = [
        (r"\bnot\s+happy\b", "sad"),
        (r"\bnot\s+feeling\s+happy\b", "sad"),
        (r"\bnot\s+feeling\s+sad\b", "nuetral"),
        (r"\bnot\s+feeling\s+angry\b", "nuetral"),
        (r"\bnot\s+angry\b", "nuetral"),
        (r"\bnot\s+surprised\b", "nuetral"),
        (r"\bnot\s+(un\w+)", lambda m: m.group(1)[2:]),  
        (r"\bnot\s+(never|no one|nothing|nowhere|none)\b", "always"),
        (r"\bnot\s+bad\b", "good"),
        (r"\bnot\s+sad\b", "happy"),
        (r"\bnot\s+angry\b", "calm"),
        (r"\bnot\s+worried\b", "relieved"),
        (r"\bnot\s+upset\b", "okay"),
        (r"\bnot\s+scared\b", "confident"),
        (r"\bnot\s+happy\b", "sad"),
        (r"\bno\s+(good|sad|angry|surprised|happy|upset)\b", "happy"),  
        (r"\bno\s+(sad)\b", "happy"),
         (r"\bnever\s+(good|bad|happy|angry|sad)\b", "good"),
         (r"\bnot\s+the\s+worst\b", "best"),
        (r"\bnot\s+feeling\s+bad\b", "feeling good"),
        (r"\bnot\s+feeling\s+down\b", "feeling up"),
        (r"\bnot\s+(un\w+)\b", lambda m: m.group(1)[2:]),
        (r"\bnot\s+(happy|unhappy)\b", "sad"),
        (r"\bnot\s+(good|bad)\b", "good"),
        (r"\bnot\s+(sad)\b", "happy"),
        (r"\bnot\s+(angry|irritated|frustrated|enraged)\b", "calm"),
        (r"\bnot\s+(scared|afraid|terrified)\b", "confident"),
        (r"\bnot\s+(gloomy)\b", "happy"),
        (r"\bnot\s+(lonely)\b", "connected"),
        (r"\bnot\s+(tired|exhausted|sleepy)\b", "energetic"),
        (r"\bnot\s+(bored)\b", "interested")
    ]

    for pattern, repl in double_negative_patterns:
        text = re.sub(pattern, repl, text, flags=re.IGNORECASE)
    
    return text

user_input = handle_double_negatives(user_input)
x_vectorizer = vectorizer.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(
     x_vectorizer, y, stratify=y, test_size=0.3, random_state=42
)
y_true = y_test
model = LogisticRegression(max_iter=1000)
model.fit(x_train,y_train)

userEmo = vectorizer.transform([user_input])
predicted_mood = model.predict(userEmo)[0]

y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)

emotion_keywords = {
    "joy": [
        "joy", "joyful", "glad", "cheerful", "content", "delighted", "excited", "promoted",
        "elated", "ecstatic", "grateful", "satisfied", "enthusiastic", "feeling on top", "optimistic", 
        "uplifted", "laughing", "giggling", "smiling", "grinning", "sunshine", "yay", "fun", "happy"
    ],
    "sadness": [
         "sad", "depressed", "down", "gloomy", "miserable", "crying", 
        "tearful", "heartbroken", "blue", "upset", "hopeless", "sorrow", 
        "lonely", "despair", "grief", "melancholy", "regret", "disappointed", "alone", "difficult", "difficulties", "stressed", "stress"
    ],
    "anger": [
        "anger","angry", "furious", "rage", "irritated", "annoyed", "frustrated", 
        "resentful", "offended", "hate", "outraged", "aggressive", "hostile", 
        "bitter", "infuriated", "pissed", "snapping", "grumpy", "enraged", "frustrating"
    ],
    "fear": [
        "scared", "afraid", "frightened", "terrified", "nervous", "anxious", 
        "worried", "panic", "tense", "shaky", "horrified", "dread", "alarmed", 
        "insecure", "paranoid", "startled", "uneasy", "timid", "phobia"
    ],
    "love": [
        "love", "loved", "loving", "affection", "passion", "caring", "adore", 
        "cherish", "fond", "romantic", "devoted", "attachment", "crush", 
        "sweetheart", "companion", "hug", "kiss", "bae", "baby", "darling"
    ],
    "surprise": [
        "surprised", "shocked", "amazed", "astonished", "stunned", "startled", 
        "speechless", "unexpected", "unbelievable", "wow", "whoa", "omg", 
        "flabbergasted", "awe", "disbelief", "jaw-dropping", "mind-blown", "shocking", "can't believe", "suprising", "surprise"
    ],
    "neutral": [
        "okay", "fine", "normal", "alright", "meh", "indifferent", "bored", 
        "blank", "neutral", "whatever", "tired", "nothing", "idle", "chill", 
        "casual", "unmoved", "plain", "still", "quiet", "calm", "lazy", "", "not feeling anything"
    ]
}


def extract_emotion_keywords(input_text, keyword_dict):
    features = {}
    for emotion, keywords in keyword_dict.items():
        features[emotion] = int(any(word in input_text.lower() for word in keywords))
    return features

user_keyword_features = extract_emotion_keywords(user_input, emotion_keywords)

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

if user_keyword_features.get(predicted_mood.replace("😊", "").replace("😢", "").replace("😠", "").replace("😨", "").replace("😮", "").replace("❤️", "").replace("😐", ""), 0) == 1:
    final_prediction = predicted_mood
else:
    matching_emotions = [emo for emo, val in user_keyword_features.items() if val == 1]
    if matching_emotions:
        final_prediction = emoji_map.get(matching_emotions[0], predicted_mood + " (uncertain)")
    else:
        final_prediction = predicted_mood

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

meal = mood_meals.get(final_prediction, "🍽️ Just stay hydrated and take a deep breath!")
meal_recipe = mood_meals_recipe.get(final_prediction)
meal_ytlink = mood_meals_ytlinks.get(final_prediction)
print(f"{final_prediction}::{meal}::{meal_recipe}::{meal_ytlink}")
print("Checking if data loads properly...", file=sys.stderr)


