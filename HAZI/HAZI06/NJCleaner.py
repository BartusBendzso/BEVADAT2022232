import pandas as pd

class NJCleaner():
    
    def __init__(self, csv_path: str) -> None:
        self.data = pd.read_csv(csv_path)

    def order_by_scheduled_time(self) -> pd.DataFrame:
        order = self.data.sort_values("scheduled_time", axis=0)
        return order

    def drop_columns_and_nan(self) -> pd.DataFrame:
        cleanedDf = self.data.drop(["from", "to"], axis=1)
        cleanedDf = self.data.dropna(axis=0)
        return cleanedDf
    
    def convert_date_to_day(self) -> pd.DataFrame:
        self.data['day'] = pd.to_datetime(self.data['day']).dt.day_name()
        convertedDf = self.data.drop('date', axis=1)
        return convertedDf

    def convert_scheduled_time_to_part_of_the_day(self) -> pd.DataFrame:
        convertDf = self.data
        convertDf['scheduled_time'] = pd.to_datetime(convertDf['scheduled_time'], format='%Y-%m-%d %H:%M:%S')
        convertDf['part_of_the_day'] = pd.cut(convertDf['scheduled_time'].dt.hour,
                                                bins=[0, 3.59, 7.59, 11.59, 15.59, 19.59, 23.59],
                                                labels=['late_night', 'early_morning', 'morning', 'afternoon', 'evening', 'night'])
        convertDf = convertDf.drop(["scheduled_time"], axis=1)
        return convertDf
    
    





    