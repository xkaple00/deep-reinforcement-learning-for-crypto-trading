import pandas as pd
import pandas_ta as ta
from datetime import datetime, timezone, timedelta
import os
import numpy as np
import pytz
from pathlib import Path
from pybit.unified_trading import HTTP
import warnings
from apscheduler.schedulers.blocking import BlockingScheduler
import keys
import argparse
import json

class BybitScraper:
    INTERVALS = {'1h': 60, '1d': 1440}
    BYBIT_INTERVALS = {'1h': 60, '1d': 'D'}
    LOOKBACK = 168 # hours
    TA_LOOKBACK = 48 # hours

    def __init__(self, coins, resolutions, start_time, end_time, save_folder, endpoint_file_paths, mode):
        self.coins = coins
        self.resolutions = resolutions
        self.start_time = start_time
        self.end_time = end_time
        self.save_folder = Path(save_folder)
        self.endpoint_file_paths = endpoint_file_paths
        self.endpoints_df = self.load_endpoints()
        self.mode = mode

        self.session = HTTP(testnet=False)
        
        self.save_folder.mkdir(parents=True, exist_ok=True)


    def load_endpoints(self):
        endpoints = {}
        for coin in self.coins:
            endpoints[coin] = {}
            for resolution in self.resolutions:
                df = pd.read_csv(self.endpoint_file_paths[coin][resolution], index_col=0)
                endpoints[coin][resolution] = df
        return endpoints
    
    def save_data_to_csv(self, df, coin, resolution, endpoint="bybit"):
        folder_path = self.save_folder / coin / resolution
        folder_path.mkdir(parents=True, exist_ok=True)
        filename = f"{endpoint}_{coin}_{resolution}_{self.file_time.strftime('%Y-%m-%d_%H:%M:%S')}.csv"
        df.to_csv(folder_path / filename)

    # # Function to map the timestamp to a point in a 2π cycle
    # def timestamp_to_week_cycle(self, timestamp, milliseconds_in_week=7*24*60*60*1000):
    #     # Calculate the total number of milliseconds since the beginning of the week (Monday)
    #     total_milliseconds = ((timestamp.dayofweek * 24 * 60 * 60 * 1000) +
    #                         (timestamp.hour * 60 * 60 * 1000) +
    #                         (timestamp.minute * 60 * 1000) +
    #                         (timestamp.second * 1000) +
    #                         timestamp.microsecond / 1000) % milliseconds_in_week
        
    #     # Map the milliseconds to a 2π cycle
    #     radians = (total_milliseconds / milliseconds_in_week) * 2 * np.pi
    #     return np.sin(radians), np.cos(radians)

    def scrape(self):
        current_time = datetime.utcnow()

        file_time = current_time.replace(second=0, microsecond=0)
        self.file_time = file_time.replace(minute=(file_time.minute // 5) * 5, second=0, microsecond=0)

        for coin in self.coins:
            for resolution in self.resolutions:
                if self.mode == "live":
                    self.end_time = current_time.replace(minute=0, second=0, microsecond=0)
                    self.start_time = self.end_time - timedelta(hours=self.LOOKBACK, minutes=self.TA_LOOKBACK*self.INTERVALS[resolution])

                self.start_time_human = self.start_time.strftime('%Y-%m-%d %H:%M:%S')
                self.end_time_human = self.end_time.strftime('%Y-%m-%d %H:%M:%S')

                print("self.start_time_human:", self.start_time_human)
                print("self.end_time_human:", self.end_time_human)

                self.start_time_unix = int(self.start_time.replace(tzinfo=timezone.utc).timestamp() * 1000)
                self.end_time_unix = int(self.end_time.replace(tzinfo=timezone.utc).timestamp() * 1000)

                print("self.start_time_human:", self.start_time_human)
                print("self.end_time_human:", self.end_time_human)

                placeholder = pd.DataFrame(index=pd.date_range(start=self.start_time_human, end=self.end_time_human, freq="1h", inclusive="left"))
                placeholder.index = placeholder.index.tz_localize(None)

                metrics = sorted(self.endpoints_df[coin][resolution]["0"].tolist())

                klines = []
                while self.start_time_unix < self.end_time_unix:
                    # print("start_time_chunk:", datetime.fromtimestamp(self.start_time_unix/1000))
                    # print("end_time_unix:", datetime.fromtimestamp(end_time_unix/1000))

                    data = self.session.get_kline(
                        category="linear",
                        symbol=coin+"USDT",
                        interval=self.BYBIT_INTERVALS[resolution],
                        start=self.start_time_unix,
                        # end=end_time_unix,
                        limit=1000
                    )        
                    
                    if not data:
                        break
                    
                    klines_chunk = data["result"]["list"][::-1]
                    klines.append(klines_chunk)
                    self.start_time_unix = int(klines_chunk[-1][0]) + self.INTERVALS[resolution] * 60 * 1000 # offset in ms

                bybit_klines_flattened = [item for sublist in klines for item in sublist]
                bybit_klines_pd = pd.DataFrame(bybit_klines_flattened, columns=['open_time', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
                bybit_klines_pd["open_time"] = pd.to_datetime(bybit_klines_pd["open_time"], unit="ms")

                klines_futures = bybit_klines_pd[['open_time', 'open', 'high', 'low', 'close']] #.astype(np.float32)

                klines_futures = klines_futures.fillna(method='ffill')
                
                # # Apply the function to the timestamp column and create new columns for the sin and cos values
                # if resolution == "1h":
                #     klines_futures['week_sin'], klines_futures['week_cos'] = zip(*klines_futures['open_time'].map(self.timestamp_to_week_cycle))

                klines_futures = klines_futures.set_index("open_time")
                klines_futures.index.name = None

                klines_futures = klines_futures.loc[(klines_futures.index >= self.start_time) & (klines_futures.index <= self.end_time)].astype(np.float32)

                # Create the Strategy
                MyStrategy = ta.Strategy(
                    name="MyStrategy",
                    ta=[{"kind": item} for item in metrics]
                )

                # Run the Strategy
                klines_futures.ta.strategy(MyStrategy)

                klines_futures.columns = [column+f"_{self.INTERVALS[resolution]:04}_{coin}" for column in klines_futures.columns.tolist()]
                # klines_futures = klines_futures.fillna(0)

                placeholder = placeholder.combine_first(klines_futures)
                placeholder = placeholder.fillna(method='ffill')

                if self.mode == "live":
                    placeholder.iloc[-self.LOOKBACK:].to_csv(os.path.join(self.save_folder, f"scraped_bybit_{coin}_{resolution}_{self.file_time.strftime('%Y-%m-%d_%H:%M:%S')}.csv"))
                
                if self.mode == "historical":
                    placeholder.to_csv(os.path.join(self.save_folder, f"scraped_bybit_{coin}_{resolution}_{self.end_time.strftime('%Y-%m-%d_%H:%M:%S')}.csv"))
                

    def run_periodic_scrape(self):
        scheduler = BlockingScheduler()
        scheduler.add_job(lambda: self.scrape(), 'cron', hour='*/1') # minute='*/5' hour='*/1'
        scheduler.start()
    

def main(args):
    save_folder_path = Path(args.save_folder)
    save_folder_path.mkdir(parents=True, exist_ok=True)

    f = open(args.endpoint_file_paths)
    endpoint_file_paths = json.load(f)

    if args.mode=="live":
        start_time = end_time = None

    if args.mode=="historical":
        start_time = datetime.fromisoformat(args.start_time)
        end_time = datetime.fromisoformat(args.end_time)

    scraper = BybitScraper(
                            coins=args.coins.split(','),
                            resolutions=args.resolutions.split(','),
                            start_time=start_time,
                            end_time=end_time,
                            endpoint_file_paths=endpoint_file_paths,
                            save_folder=save_folder_path,
                            mode=args.mode,
                            )

    warnings.filterwarnings("ignore")

    if args.mode=="live":
        scraper.run_periodic_scrape()

    if args.mode=="historical":
        scraper.scrape()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the Bybit data scraper with specified parameters.')

    parser.add_argument('--coins', required=True, help='Comma-separated list of coin names, e.g.,"BTC,ETH,FTM,BAT"')
    parser.add_argument('--resolutions', required=True, help='Comma-separated list of resolutions, e.g.,"1h,1d"')
    parser.add_argument('--start_time', required=False, help='Start datetime in ISO format, e.g., "2020-07-01T00:00:00", not used for live data')
    parser.add_argument('--end_time', required=False, help='End datetime in ISO format, e.g., "2024-04-11T00:00:00", not used for live data')
    parser.add_argument('--endpoint_file_paths', required=True, help='path of endpoints_file_path_bybit.json')
    parser.add_argument('--save_folder', required=True, help='Folder path to save the scraped data, e.g., "./data/test/bybit/historical"')
    parser.add_argument('--mode', required=True, help='mode to scrape data', choices=["historical", "live"])

    args = parser.parse_args()

    main(args)