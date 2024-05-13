import pandas as pd
import pandas_ta as ta
from datetime import datetime, timezone, timedelta
import os
from binance import Client
import numpy as np
import pytz
from pathlib import Path
import warnings
from apscheduler.schedulers.blocking import BlockingScheduler
import keys
import argparse
import json

class BinanceScraper:
    INTERVALS = {'1h': 60, '1d': 1440}
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
        self.client = Client()
        self.mode = mode

    def load_endpoints(self):
        endpoints = {}
        for coin in self.coins:
            endpoints[coin] = {}
            for resolution in self.resolutions:
                df = pd.read_csv(self.endpoint_file_paths[coin][resolution], index_col=0)
                endpoints[coin][resolution] = df
        return endpoints
    
    def save_data_to_csv(self, df, coin, resolution, endpoint="binance"):
        folder_path = self.save_folder / coin / resolution
        folder_path.mkdir(parents=True, exist_ok=True)
        filename = f"{endpoint}_{coin}_{resolution}_{self.file_time.strftime('%Y-%m-%d_%H:%M:%S')}.csv"
        df.to_csv(folder_path / filename)

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

                placeholder = pd.DataFrame(index=pd.date_range(start=self.start_time_human, end=self.end_time_human, freq="1h", inclusive="left"))
                placeholder.index = placeholder.index.tz_localize(None)

                metrics = sorted(self.endpoints_df[coin][resolution]["0"].tolist())

                # klines_futures = self.client.get_historical_klines(symbol=coin+"USDT", interval=resolution, start_str=self.start_time_human, end_str=self.end_time_human)
                klines_futures = self.client.futures_historical_klines(symbol=coin+"USDT", interval=resolution, start_str=self.start_time_human, end_str=self.end_time_human)

                klines_futures = pd.DataFrame(klines_futures, columns=['open_time','open', 'high', 'low', 'close', 'volume', 'close_time', 'qav','num_trades','taker_base_vol', 'taker_quote_vol', 'ignore'])
                klines_futures = klines_futures[['open_time','open', 'high', 'low', 'close']].astype(np.float64) # .astype(np.float64) is a must

                if self.mode == "live":
                    klines_futures = klines_futures.iloc[:-1] # ignore last not closed candle

                klines_futures = klines_futures.fillna(method='ffill')
                
                klines_futures['open_time'] = pd.to_datetime(klines_futures['open_time'], unit='ms')

                klines_futures = klines_futures.set_index("open_time")
                klines_futures.index.name = None

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
                    placeholder.iloc[-self.LOOKBACK:].to_csv(os.path.join(self.save_folder, f"scraped_binance_{coin}_{resolution}_{self.file_time.strftime('%Y-%m-%d_%H:%M:%S')}.csv"))
                
                if self.mode == "historical":
                    placeholder.to_csv(os.path.join(self.save_folder, f"scraped_binance_{coin}_{resolution}_{self.end_time.strftime('%Y-%m-%d_%H:%M:%S')}.csv"))
                

    def run_periodic_scrape(self):
        scheduler = BlockingScheduler()
        scheduler.add_job(lambda: self.scrape(), 'cron',  hour='*/1') # minute='*/1' hour='*/1'
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
    
    scraper = BinanceScraper(
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
    parser = argparse.ArgumentParser(description='Run the Binance data scraper with specified parameters.')
    parser.add_argument('--coins', required=True, help='Comma-separated list of coin names, e.g.,"BTC,ETH,FTM,BAT"')
    parser.add_argument('--resolutions', required=True, help='Comma-separated list of resolutions, e.g.,"1h,1d"')
    parser.add_argument('--start_time', required=False, help='Start datetime in ISO format, e.g., "2020-07-01T00:00:00", not used for live data')
    parser.add_argument('--end_time', required=False, help='End datetime in ISO format, e.g., "2024-04-11T00:00:00", not used for live data')
    parser.add_argument('--endpoint_file_paths', required=True, help='path of endpoints_file_path_binance.json')
    parser.add_argument('--save_folder', required=True, help='Folder path to save the scraped data, e.g., "./data/test/binance/historical"')
    parser.add_argument('--mode', required=True, help='mode to scrape data', choices=["historical", "live"])

    args = parser.parse_args()
    main(args)