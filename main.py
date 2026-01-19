import pandas as pd
import seaborn as sns

# Загружаем стандартный датасет 'titanic' из библиотеки seaborn
titanic_data = sns.load_dataset('titanic')

# Выбираем только нужные столбцы: класс каюты, пол и возраст
selected_features = titanic_data[['pclass', 'sex', 'age']].copy()

# Сохраняем этот поднабор данных в CSV-файл без индекса
selected_features.to_csv('titanic_subset.csv', index=False)

# Заполняем пропущенные значения в столбце 'age' средним по имеющимся данным
selected_features['age'] = selected_features['age'].fillna(selected_features['age'].mean())

# Преобразуем категориальный признак 'sex' в числовой формат с помощью one-hot encoding,
# оставляя только один столбец (drop_first=True)
encoded_data = pd.get_dummies(selected_features, columns=['sex'], drop_first=True)

# Сохраняем обработанный датасет в новый CSV-файл
encoded_data.to_csv('titanic_subset_updated.csv', index=False)

# Выводим подтверждение и первые строки результата
print("Файлы с датасетами успешно сохранены.")
print(encoded_data.head())
