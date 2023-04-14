import pandas as pd
import math as ma

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
        convertedDf = self.data
        convertedDf['day'] = pd.to_datetime(convertedDf['day']).dt.day_name()
        convertedDf = convertedDf.drop('date', axis=1)
        return convertedDf

    def convert_scheduled_time_to_part_of_the_day(self) -> pd.DataFrame:
        convertDf = self.data
        convertDf['scheduled_time'] = pd.to_datetime(convertDf['scheduled_time'], format='%Y-%m-%d %H:%M:%S')
        convertDf['part_of_the_day'] = pd.cut(convertDf['scheduled_time'].dt.hour,
                                                bins=[0, 3.59, 7.59, 11.59, 15.59, 19.59, 23.59],
                                                labels=['late_night', 'early_morning', 'morning', 'afternoon', 'evening', 'night'])
        convertDf = convertDf.drop(["scheduled_time"], axis=1)
        return convertDf
    

    def convert_delay(self) -> pd.DataFrame:
        delayDf = self.data
        delayDf['delay'] = pd.cut(delayDf['delay_minutes'],
                                bins=[-1, 5, ma.inf],
                                labels=[0, 1])
        return delayDf
    
    def drop_unnecessary_columns(self) -> pd.DataFrame:
        dropColDf = self.data.drop(['train_id', 'actual_time', 'delay_minutes'],
                                   axis=1)
        return dropColDf
    
    def save_first_60k(self, save_path: str):
        save_df = self.data.head(60000)
        return save_df.to_csv(path_or_buf=save_path, index=False)
    
    def prep_df(self, save_path: str = 'data/NJ.csv'):
        self.data = (self.order_by_scheduled_time(),
                    self.drop_columns_and_nan(),
                    self.convert_date_to_day(),
                    self.convert_scheduled_time_to_part_of_the_day(),
                    self.convert_delay(),
                    self.drop_unnecessary_columns(),
                    self.save_first_60k(save_path))

    







    