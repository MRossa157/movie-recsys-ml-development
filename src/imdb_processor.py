import pandas as pd


class IMDBDataProcessor:
    def __init__(self, imdb_files: dict[str, str], low_memory: bool = True):
        """
        Инициализирует процессор IMDb файлов.

        Args:
            imdb_files (dict): Словарь с путями к IMDb файлам.
                Пример:
                {
                    'akas': 'path/to/title.akas.tsv',
                    'basics': 'path/to/title.basics.tsv',
                    'crew': 'path/to/title.crew.tsv',
                    'episode': 'path/to/title.episode.tsv',
                    'principals': 'path/to/title.principals.tsv',
                    'ratings': 'path/to/title.ratings.tsv',
                    'names': 'path/to/name.basics.tsv',
                }
        """
        self.files = imdb_files
        self.low_memory = low_memory

    def load_file(self, file_path, sep='\t'):
        """Загружает TSV файл с обработкой '\\N' как NaN."""
        return pd.read_csv(
            file_path,
            sep=sep,
            na_values='\\N',
            low_memory=self.low_memory,
        )

    def process(self):
        """Обрабатывает и объединяет данные из всех файлов."""
        # Загрузка всех файлов
        akas = self.load_file(self.files['akas'])
        basics = self.load_file(self.files['basics'])
        crew = self.load_file(self.files['crew'])
        episode = self.load_file(self.files['episode'])
        principals = self.load_file(self.files['principals'])
        ratings = self.load_file(self.files['ratings'])
        names = self.load_file(self.files['names'])

        # Обработка akas
        akas = akas[
            ['titleId', 'title', 'region', 'language', 'isOriginalTitle']
        ].rename(columns={'titleId': 'tconst'})

        # Обработка basics
        basics = basics[
            [
                'tconst',
                'titleType',
                'primaryTitle',
                'originalTitle',
                'isAdult',
                'startYear',
                'runtimeMinutes',
                'genres',
            ]
        ]

        # Обработка crew
        crew = crew[['tconst', 'directors', 'writers']]

        # Обработка episode
        episode = episode[
            ['tconst', 'parentTconst', 'seasonNumber', 'episodeNumber']
        ]

        # Обработка principals
        principals = principals[
            ['tconst', 'nconst', 'category', 'job', 'characters']
        ]

        # Обработка ratings
        ratings = ratings[['tconst', 'averageRating', 'numVotes']]

        # Объединение данных
        merged = basics.merge(akas, on='tconst', how='left')
        merged = merged.merge(crew, on='tconst', how='left')
        merged = merged.merge(episode, on='tconst', how='left')
        merged = merged.merge(ratings, on='tconst', how='left')

        # Возвращаем итоговый DataFrame
        return merged


# Пример использования
if __name__ == '__main__':
    import time
    imdb_files = {
        'akas': r'src/datasets/imdb/title.akas.tsv',
        'basics': r'src/datasets/imdb/title.basics.tsv',
        'crew': r'src/datasets/imdb/title.crew.tsv',
        'episode': r'src/datasets/imdb/title.episode.tsv',
        'principals': r'src/datasets/imdb/title.principals.tsv',
        'ratings': r'src/datasets/imdb/title.ratings.tsv',
        'names': r'src/datasets/imdb/name.basics.tsv',
    }

    processor = IMDBDataProcessor(imdb_files, low_memory=False)
    start_time = time.time()
    imdb_metadata = processor.process()
    print(time.time() - start_time)

    # Вывод первых строк итогового DataFrame
    print(imdb_metadata.head())
