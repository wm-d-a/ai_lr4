import random
import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy
from collections import defaultdict
from surprise import KNNWithMeans


def get_top_n(predictions, n=10):
    """Return the top-N recommendation for each user from a set of predictions.

    Args:
        predictions(list of Prediction objects): The list of predictions, as
            returned by the test method of an algorithm.
        n(int): The number of recommendation to output for each user. Default
            is 10.

    Returns:
    A dict where keys are user (raw) ids and values are lists of tuples:
        [(raw item id, rating estimation), ...] of size n.
    """

    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n


# Создаем список пользователей и книг
# users = [str(i) for i in range(1, 101)]
# books = [str(i) for i in range(1, 201)]

# Генерируем случайные оценки пользователей для книг
# data = []
# for user in users:
#     for book in books:
#         rating = random.randint(1, 5)
#         data.append([user, book, rating])

# Создаем DataFrame из данных
# df = pd.DataFrame(data, columns=['userId', 'movieId', 'rating'])
df = pd.read_csv('./ml-20m/ratings.csv', usecols=['userId', 'movieId', 'rating'], nrows=10000)
# Создаем объект Reader для определения формата данных
reader = Reader(rating_scale=(1, 5))
users = df['userId'].unique()
books = df['movieId'].unique()
# print(len(users), len(books))

# Создаем датасет из DataFrame и объекта Reader
dataset = Dataset.load_from_df(df[['userId', 'movieId', 'rating']], reader)

trainset, testset = train_test_split(dataset, test_size=0.2, random_state=42)

# To use item-based cosine similarity
sim_options = {
    "name": "cosine",
    "user_based": True,  # Compute  similarities between items
}
model = KNNWithMeans(sim_options=sim_options)
# model = SVD()
model.fit(trainset)

predictions = model.test(testset)

# mae = accuracy.mae(predictions)
# rmse = accuracy.rmse(predictions)

# print(f'MAE: {mae}')
# print(f'RMSE: {rmse}')

# Выбираем случайного пользователя
userId = 91  # random.choice(users)

# Получаем топ N рекомендаций для пользователя
top_n = get_top_n(predictions, n=10)

print(f"Рекомендации книг для пользователя {userId}:")
for movieId, rating in top_n[userId]:
    print(f"Книга ID: {movieId}, Рейтинг: {rating}")
