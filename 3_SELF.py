import pandas as pd
from pandasql import sqldf


def get_avg_spends_for_user(dataset, user):
    result = dataset.groupby('User')['Price'].mean()
    return result[user]


def get_most_popular_category_for_user(dataset, user):
    result = sqldf(
        f'''
    SELECT
        Category,
        max(ct) as mx
    FROM
    (
        SELECT 
            Category,
            count(*) as ct
        FROM dataset 
        WHERE User = '{user}'
        GROUP BY Category
    ) as v0
    --where ct = max(ct)
    ''')
    return result['Category'][0]


def get_top_item_per_user(price, category, items_dataset):
    top3_items = sqldf(
        f'''
        SELECT
            Item,
            Price,
            Rating,
            delta
        FROM 
        (
            SELECT 
                Item,
                Price,
                abs(Price - {price}) as delta,
                Rating 
            FROM items_dataset 
            WHERE Category = '{category}'
        ) as v0
        ORDER BY Rating DESC, delta ASC
        LIMIT 3
        ''')
    result = [[top3_items['Item'][0], top3_items['Rating'][0], top3_items['Price'][0], top3_items['delta'][0]],
              [top3_items['Item'][1], top3_items['Rating'][1], top3_items['Price'][1], top3_items['delta'][1]],
              [top3_items['Item'][2], top3_items['Rating'][2], top3_items['Price'][2], top3_items['delta'][2]]]
    return result


def get_top_items(user_meta, items_dataset):
    result = {}
    # print(user_meta)
    for user in user_meta:
        top_items = []
        top_items = get_top_item_per_user(user_meta[user]['avg_spend'], user_meta[user]['top_category'], items_dataset)
        result[user] = top_items
    # print(result)
    return result


# get data
spends_dataset = pd.read_excel('example.xlsx', sheet_name='Spends')
items_dataset = pd.read_excel('example.xlsx', sheet_name='Items')

unique_users = spends_dataset['User'].unique()

user_meta = {}
# print(spends_dataset[spends_dataset['User'] == 'Ivan'])
for user in unique_users:
    avg_spend = get_avg_spends_for_user(spends_dataset, user)
    top_category = get_most_popular_category_for_user(spends_dataset, user)
    user_meta[user] = {'avg_spend': avg_spend, 'top_category': top_category}
    # print(user, avg_spend, top_category, '\n')

result = get_top_items(user_meta, items_dataset)
user_name = "Yana"
# print()
# print(result[user_name], user_meta[user_name]['avg_spend'])
print(f'Рекомендуемые товары для пользователя {user_name.capitalize()}')
for recommendations in result[user_name]:
    print(f'Товар: {recommendations[0]}; Цена: {recommendations[2]}; Рейтинг товара: {recommendations[1]}')
# print(result['Ortem'], user_meta['Ortem']['avg_spend'])
'''
Выдача рекомендуемых товаров для юзера с учетом среднего чека, частых категорий, и рейтинга товара
'''
